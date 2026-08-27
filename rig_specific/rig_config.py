"""
Per-user rig configuration for the move-and-attach pipeline.

Each user keeps a small config file describing where their data comes from and where it
should land, so one script serves everybody instead of a separate copy per person.

Credentials are deliberately NOT part of the config. A config that contains a password is
rejected, because these files are meant to live in the repo next to the code. Passwords are
prompted for at run time, or read from an environment variable, or avoided entirely by using
an SSH key or agent.

Config format is JSON, or YAML when PyYAML happens to be installed (it is not a dependency).

Lookup order for `--config`:
    1. an explicit path, if it exists
    2. <this directory>/configs/<name>.json  (or .yaml/.yml)
    3. ~/.config/visanalysis/<name>.json     (or .yaml/.yml)
With no --config: $VISANALYSIS_RIG_CONFIG, else ~/.config/visanalysis/rig_config.json.

https://github.com/ClandininLab/visanalysis
"""
import getpass
import json
import os


CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs')
USER_CONFIG_DIR = os.path.expanduser('~/.config/visanalysis')
CONFIG_SUFFIXES = ('.json', '.yaml', '.yml')

# Keys that must never appear in a config file. Anything matching is a hard error rather than
# a warning, so a secret cannot be committed by accident.
FORBIDDEN_KEYS = ('password', 'passwd', 'secret', 'token', 'private_key', 'passphrase')


class ConfigError(RuntimeError):
    """The config is missing, malformed, or contains something it must not."""


def _loadFile(path):
    with open(path, 'r') as fh:
        text = fh.read()
    if path.endswith(('.yaml', '.yml')):
        try:
            import yaml
        except ImportError:
            raise ConfigError(
                '{} is YAML but PyYAML is not installed. Either `pip install pyyaml` or '
                'convert the config to JSON.'.format(path))
        return yaml.safe_load(text)
    try:
        return json.loads(text)
    except ValueError as err:
        raise ConfigError('{} is not valid JSON: {}'.format(path, err))


def findConfig(name_or_path=None):
    """Resolve a config name or path to a file on disk."""
    if name_or_path:
        if os.path.isfile(name_or_path):
            return os.path.abspath(name_or_path)
        for directory in (CONFIG_DIR, USER_CONFIG_DIR):
            for suffix in CONFIG_SUFFIXES:
                candidate = os.path.join(directory, name_or_path + suffix)
                if os.path.isfile(candidate):
                    return candidate
        raise ConfigError(
            "No config named {!r}. Looked for it as a path, and as <name>{{{}}} in:\n"
            '  {}\n  {}\nAvailable: {}'.format(
                name_or_path, ','.join(CONFIG_SUFFIXES), CONFIG_DIR, USER_CONFIG_DIR,
                ', '.join(listConfigs()) or '(none)'))

    env_path = os.environ.get('VISANALYSIS_RIG_CONFIG')
    if env_path:
        if not os.path.isfile(env_path):
            raise ConfigError(
                'VISANALYSIS_RIG_CONFIG points at {}, which does not exist.'.format(env_path))
        return os.path.abspath(env_path)

    for suffix in CONFIG_SUFFIXES:
        candidate = os.path.join(USER_CONFIG_DIR, 'rig_config' + suffix)
        if os.path.isfile(candidate):
            return candidate

    raise ConfigError(
        'No config given and no default found. Pass --config <name-or-path>, set '
        '$VISANALYSIS_RIG_CONFIG, or create {}/rig_config.json.\nAvailable configs: {}'.format(
            USER_CONFIG_DIR, ', '.join(listConfigs()) or '(none)'))


def listConfigs():
    """Names of every config discoverable without an explicit path."""
    names = set()
    for directory in (CONFIG_DIR, USER_CONFIG_DIR):
        if not os.path.isdir(directory):
            continue
        for entry in os.listdir(directory):
            stem, suffix = os.path.splitext(entry)
            if suffix in CONFIG_SUFFIXES and not stem.startswith('_'):
                names.add(stem)
    return sorted(names)


def _expand(path):
    """Expand ~ and $VARS so configs stay portable between machines."""
    return os.path.expanduser(os.path.expandvars(path)) if path else path


def _assertNoSecrets(node, path='config'):
    """Refuse a config that carries a credential, at any depth."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key.lower() in FORBIDDEN_KEYS:
                raise ConfigError(
                    "{}.{} is a credential and must not live in a config file. Configs sit "
                    'next to the code and get committed. Use an SSH key, an agent, '
                    '{{\"auth\": {{\"method\": \"env\", \"var\": \"...\"}}}}, or the default '
                    'interactive prompt instead.'.format(path, key))
            _assertNoSecrets(value, '{}.{}'.format(path, key))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            _assertNoSecrets(value, '{}[{}]'.format(path, i))


class RigConfig:
    """A validated per-user configuration."""

    def __init__(self, payload, path=None):
        self.path = path
        self.raw = payload
        _assertNoSecrets(payload)

        if not isinstance(payload, dict):
            raise ConfigError('{}: top level must be an object.'.format(path))

        self.name = payload.get('name') or (
            os.path.splitext(os.path.basename(path))[0] if path else 'unnamed')
        self.plugin = payload.get('plugin', 'fortyhourfitness')

        target = payload.get('target')
        if not target or not target.get('path'):
            raise ConfigError('{}: "target.path" is required.'.format(path))
        self.target_path = _expand(target['path'])
        self.target_date_format = target.get('date_format', '%Y-%m-%d')
        permissions = target.get('permissions')
        self.permissions = int(str(permissions), 8) if permissions is not None else None

        self.sources = []
        for i, source in enumerate(payload.get('sources', [])):
            self.sources.append(_Source(source, i, path))
        if not self.sources:
            raise ConfigError('{}: at least one entry in "sources" is required.'.format(path))

        self.attach_kwargs = dict(payload.get('attach', {}))

    def targetDir(self, date):
        return os.path.join(self.target_path, formatDate(date, self.target_date_format))

    def describe(self):
        lines = ['config: {}{}'.format(self.name, ' ({})'.format(self.path) if self.path else ''),
                 '  plugin: {}'.format(self.plugin),
                 '  target: {}  (date dirs as {})'.format(self.target_path, self.target_date_format)]
        if self.permissions is not None:
            lines.append('  permissions: {:o}'.format(self.permissions))
        for source in self.sources:
            lines.append('  source {}'.format(source.describe()))
        if self.attach_kwargs:
            lines.append('  attach defaults: {}'.format(self.attach_kwargs))
        return '\n'.join(lines)


class _Source:
    """One place data is pulled from: a local/mounted path, or an SFTP host."""

    def __init__(self, payload, index, config_path):
        self.name = payload.get('name', 'source{}'.format(index))
        self.type = payload.get('type', 'local')
        if self.type not in ('local', 'sftp'):
            raise ConfigError('{}: source {!r} has unknown type {!r}; expected "local" or '
                              '"sftp".'.format(config_path, self.name, self.type))
        if not payload.get('path'):
            raise ConfigError('{}: source {!r} needs a "path".'.format(config_path, self.name))
        self.path = _expand(payload['path'])
        # Sources often disagree on how the date directory is spelled - the Jackfish/assist
        # side writes 20260819 while the rig side writes 2026-08-19 - so this is per source.
        self.date_format = payload.get('date_format', '%Y-%m-%d')
        self.required = bool(payload.get('required', True))

        self.host = payload.get('host')
        self.port = int(payload.get('port', 22))
        self.username = payload.get('username') or getpass.getuser()
        self.auth = dict(payload.get('auth', {'method': 'prompt'}))
        method = self.auth.get('method', 'prompt')
        if method not in ('prompt', 'env', 'key', 'agent'):
            raise ConfigError(
                '{}: source {!r} has unknown auth method {!r}; expected prompt, env, key or '
                'agent.'.format(config_path, self.name, method))
        if method == 'env' and not self.auth.get('var'):
            raise ConfigError('{}: source {!r} uses auth method "env" but names no '
                              '"var".'.format(config_path, self.name))
        if method == 'key' and not self.auth.get('key_filename'):
            raise ConfigError('{}: source {!r} uses auth method "key" but names no '
                              '"key_filename".'.format(config_path, self.name))
        if self.type == 'sftp' and not self.host:
            raise ConfigError('{}: sftp source {!r} needs a "host".'.format(config_path, self.name))

    def sourceDir(self, date):
        return os.path.join(self.path, formatDate(date, self.date_format))

    def describe(self):
        if self.type == 'local':
            return '{} [local] {}  (date dirs as {}){}'.format(
                self.name, self.path, self.date_format, '' if self.required else '  [optional]')
        return '{} [sftp] {}@{}:{}  (date dirs as {}, auth={}){}'.format(
            self.name, self.username, self.host, self.path, self.date_format,
            self.auth.get('method', 'prompt'), '' if self.required else '  [optional]')


def formatDate(date, date_format):
    """Render an ISO date string in a source's own convention.

    Accepts YYYY-MM-DD and reformats it; anything else is passed through untouched, so a user
    whose directories are not date-shaped at all can still name them directly.
    """
    import datetime
    if date_format in (None, '', '%Y-%m-%d'):
        return date
    try:
        parsed = datetime.datetime.strptime(date, '%Y-%m-%d')
    except ValueError:
        return date
    return parsed.strftime(date_format)


def loadConfig(name_or_path=None):
    path = findConfig(name_or_path)
    return RigConfig(_loadFile(path), path)


# ------------------------------------------------------------------------- #
# Credentials. Never read from, or written to, the config file.
# ------------------------------------------------------------------------- #

_PASSWORD_CACHE = {}


def resolveAuth(source, interactive=True):
    """Work out how to authenticate to an SFTP source.

    returns a dict of paramiko connect kwargs: either {'password': ...} or
    {'key_filename': ...}, or {} to fall through to the SSH agent.
    """
    method = source.auth.get('method', 'prompt')

    if method == 'agent':
        return {}

    if method == 'key':
        key_filename = _expand(source.auth['key_filename'])
        if not os.path.isfile(key_filename):
            raise ConfigError('SSH key {} for source {!r} does not exist.'.format(
                key_filename, source.name))
        return {'key_filename': key_filename}

    if method == 'env':
        var = source.auth['var']
        password = os.environ.get(var)
        if not password:
            raise ConfigError(
                'Source {!r} expects the password in ${}, which is unset or empty.'.format(
                    source.name, var))
        return {'password': password}

    # method == 'prompt'
    cache_key = (source.host, source.port, source.username)
    if cache_key in _PASSWORD_CACHE:
        return {'password': _PASSWORD_CACHE[cache_key]}
    if not interactive:
        raise ConfigError(
            'Source {!r} needs an interactive password prompt, but this run is '
            'non-interactive. Use auth method "env", "key" or "agent" instead.'.format(source.name))
    password = getpass.getpass('Password for {}@{}: '.format(source.username, source.host))
    if not password:
        raise ConfigError('No password entered for {}@{}.'.format(source.username, source.host))
    _PASSWORD_CACHE[cache_key] = password
    return {'password': password}


def clearCachedPasswords():
    """Drop any password held in memory for this run."""
    _PASSWORD_CACHE.clear()
