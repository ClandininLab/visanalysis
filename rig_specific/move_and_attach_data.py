#!/usr/bin/env python3
"""
Pull a day's acquisition data together from its sources, then attach it to the stimpack hdf5.

Generic replacement for the per-user move_and_attach_data_*.py scripts. Everything that used
to be hard-coded - source and target directories, host, username, date-directory spelling,
which plugin to use - now lives in a per-user config file; see rig_config.py and
configs/example.json. Passwords are prompted for, never stored.

Usage:
    python move_and_attach_data.py --config michelle_fortyhr 2026-08-19
    python move_and_attach_data.py --config ./my_config.json 2026-08-19 2026-08-20
    python move_and_attach_data.py --list-configs
    python move_and_attach_data.py --config michelle_fortyhr --dry-run 2026-08-19

Transfer options:
    --skip-transfer     attach only; do not copy anything
    --transfer-only     copy only; do not attach
    --dry-run           print what would happen, touch nothing
    --non-interactive   never prompt (auth must be env/key/agent)

Attach options (passed through to the plugin, when it accepts them):
    --no-ensemble       never split a .jfdaqdata; one block per file
    --require-ensemble  fail if any .jfdaqdata yields only a single block
    --counts-only       allow a cut from FicTrac line counts alone, when no camera stopped
    --force             downgrade every failing validation check to a warning
    --no-verify         skip the photodiode audit
    --ensemble-json P   path to an ensemble.json override

https://github.com/ClandininLab/visanalysis
"""
import argparse
import inspect
import os
import shutil
import sys
from stat import S_ISDIR

# Allow running straight out of a checkout, without installing the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rig_config import (ConfigError, clearCachedPasswords, listConfigs, loadConfig,  # noqa: E402
                        resolveAuth)


# --------------------------------------------------------------------------- #
# Plugins
# --------------------------------------------------------------------------- #

def getPlugin(name):
    """Instantiate an acquisition plugin by config name."""
    key = name.lower().replace('_', '').replace('-', '')
    if key in ('fortyhourfitness', '40hourfitness', 'fortyhr', '40hr'):
        from visanalysis.plugin.fortyhourfitness import FortyHourFitnessPlugin
        return FortyHourFitnessPlugin()
    if key in ('twentyfourhourfitness', '24hourfitness', 'twentyfourhr', '24hr'):
        from visanalysis.plugin.twentyfourhourfitness import TwentyFourHourFitnessPlugin
        return TwentyFourHourFitnessPlugin()
    if key == 'bruker':
        from visanalysis.plugin.bruker import BrukerPlugin
        return BrukerPlugin()
    if key in ('aodscope', 'aod'):
        from visanalysis.plugin.aodscope import AodScopePlugin
        return AodScopePlugin()
    raise ConfigError(
        'Unknown plugin {!r}. Known: fortyhourfitness, twentyfourhourfitness, bruker, '
        'aodscope.'.format(name))


# --------------------------------------------------------------------------- #
# Transfer
# --------------------------------------------------------------------------- #

def sftp_get_recursive(path, dest, sftp):
    item_list = sftp.listdir_attr(path)
    dest = str(dest)
    if not os.path.isdir(dest):
        os.makedirs(dest, exist_ok=True)
    for item in item_list:
        mode = item.st_mode
        if S_ISDIR(mode):
            sftp_get_recursive(os.path.join(path, item.filename), os.path.join(dest, item.filename), sftp)
        else:
            sftp.get(os.path.join(path, item.filename), os.path.join(dest, item.filename))


def change_permission_recursive(path, mode=0o774):
    for root, dirs, files in os.walk(path):
        for dir in [os.path.join(root, d) for d in dirs]:
            os.chmod(dir, mode)
        for file in [os.path.join(root, f) for f in files]:
            os.chmod(file, mode)


def copyLocalSource(source, date, target_dir, dry_run=False):
    src = source.sourceDir(date)
    if not os.path.isdir(src):
        message = 'source {!r}: {} does not exist'.format(source.name, src)
        if source.required:
            raise FileNotFoundError(message)
        print('  SKIP  {} (optional)'.format(message))
        return False
    print('  COPY  {}  ->  {}'.format(src, target_dir))
    if not dry_run:
        shutil.copytree(src, target_dir, dirs_exist_ok=True)
    return True


def copySftpSource(source, date, target_dir, dry_run=False, interactive=True):
    try:
        import paramiko
    except ImportError:
        raise ConfigError('source {!r} is sftp but paramiko is not installed '
                          '(`pip install paramiko`).'.format(source.name))

    src = source.sourceDir(date)
    print('  COPY  {}@{}:{}  ->  {}'.format(source.username, source.host, src, target_dir))
    if dry_run:
        return True

    connect_kwargs = resolveAuth(source, interactive=interactive)
    transport = None
    try:
        if 'key_filename' in connect_kwargs or not connect_kwargs:
            # Key and agent auth go through SSHClient, which knows how to use both.
            client = paramiko.SSHClient()
            client.load_system_host_keys()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(source.host, port=source.port, username=source.username,
                           allow_agent=True, look_for_keys=True, **connect_kwargs)
            with client.open_sftp() as sftp:
                sftp_get_recursive(src, target_dir, sftp)
            client.close()
        else:
            transport = paramiko.Transport((source.host, source.port))
            transport.connect(username=source.username, password=connect_kwargs['password'])
            with paramiko.SFTPClient.from_transport(transport) as sftp:
                sftp_get_recursive(src, target_dir, sftp)
    except Exception as err:
        # repr() of a paramiko exception can echo connection detail but never the password;
        # still, report the type and message rather than the whole object graph.
        message = 'source {!r}: {}: {}'.format(source.name, type(err).__name__, err)
        if source.required:
            raise RuntimeError(message)
        print('  WARN  {} (optional source, continuing)'.format(message))
        return False
    finally:
        if transport is not None:
            transport.close()
    return True


def gatherDate(config, date, dry_run=False, interactive=True):
    """Copy every configured source for one date into the target date directory."""
    target_dir = config.targetDir(date)
    print('  target: {}'.format(target_dir))
    if not dry_run:
        os.makedirs(target_dir, exist_ok=True)

    for source in config.sources:
        if source.type == 'local':
            copyLocalSource(source, date, target_dir, dry_run=dry_run)
        else:
            copySftpSource(source, date, target_dir, dry_run=dry_run, interactive=interactive)

    if config.permissions is not None and not dry_run:
        try:
            change_permission_recursive(target_dir, config.permissions)
        except OSError as err:
            print('  WARN  failed changing permissions on {}: {}'.format(target_dir, err))
    return target_dir


# --------------------------------------------------------------------------- #
# Attach
# --------------------------------------------------------------------------- #

def attachDate(config, date_dir, attach_kwargs):
    hdf5_files = sorted(x for x in os.listdir(date_dir) if x.endswith('.hdf5'))
    if len(hdf5_files) != 1:
        raise RuntimeError('expected exactly one .hdf5 in {}, found {}'.format(
            date_dir, hdf5_files))

    file_path = os.path.join(date_dir, hdf5_files[0])
    plugin = getPlugin(config.plugin)

    # Only forward the options this plugin actually accepts, so a config written for the
    # 40-hour rig does not blow up when pointed at bruker or aodscope.
    accepted = inspect.signature(plugin.attachData).parameters
    supported = {k: v for k, v in attach_kwargs.items() if k in accepted}
    dropped = sorted(set(attach_kwargs) - set(supported))
    if dropped:
        print('  NOTE  {} does not accept {}; ignoring.'.format(config.plugin, dropped))
    if 'report_path' in accepted:
        supported.setdefault('report_path', os.path.join(
            date_dir, 'attach_report_{}.json'.format(hdf5_files[0][:-5])))

    print('  attaching {} with the {} plugin'.format(file_path, config.plugin))
    return plugin.attachData(experiment_file_name=None, file_path=file_path,
                             data_directory=date_dir, **supported)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def buildParser():
    parser = argparse.ArgumentParser(
        description='Move acquisition data into place and attach it to the stimpack hdf5.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Configs are per user; see configs/example.json. Passwords are never stored '
               'in a config - they are prompted for, or taken from an env var / SSH key.')
    parser.add_argument('dates', nargs='*', metavar='YYYY-MM-DD')
    parser.add_argument('--config', '-c', help='config name or path')
    parser.add_argument('--list-configs', action='store_true', help='list configs and exit')
    parser.add_argument('--show-config', action='store_true', help='print the config and exit')
    parser.add_argument('--dry-run', action='store_true', help='print actions, change nothing')
    parser.add_argument('--skip-transfer', action='store_true', help='attach only')
    parser.add_argument('--transfer-only', action='store_true', help='copy only')
    parser.add_argument('--non-interactive', action='store_true', help='never prompt')

    attach = parser.add_argument_group('attach options')
    attach.add_argument('--no-ensemble', dest='ensemble', action='store_const', const='off')
    attach.add_argument('--require-ensemble', dest='ensemble', action='store_const', const='require')
    attach.add_argument('--counts-only', dest='allow_counts_only', action='store_true', default=None)
    attach.add_argument('--force', dest='strict', action='store_false', default=None)
    attach.add_argument('--no-verify', dest='verify_photodiode', action='store_false', default=None)
    attach.add_argument('--ensemble-json', dest='ensemble_override', metavar='PATH')
    return parser


def main(argv=None):
    args = buildParser().parse_args(argv)

    if args.list_configs:
        names = listConfigs()
        print('\n'.join(names) if names else 'No configs found.')
        return 0

    config = loadConfig(args.config)

    if args.show_config:
        print(config.describe())
        return 0

    if not args.dates:
        raise SystemExit('No dates given. Example:\n  python move_and_attach_data.py '
                         '--config {} 2026-08-19'.format(config.name))

    # CLI flags win over the config's attach defaults.
    attach_kwargs = dict(config.attach_kwargs)
    for key in ('ensemble', 'allow_counts_only', 'strict', 'verify_photodiode',
                'ensemble_override'):
        value = getattr(args, key, None)
        if value is not None:
            attach_kwargs[key] = value

    print(config.describe())
    print('dates: {}'.format(', '.join(args.dates)))
    print('attach options: {}'.format(attach_kwargs or '(plugin defaults)'))
    if args.dry_run:
        print('*** DRY RUN - nothing will be copied, written, or attached ***')

    failures = []
    try:
        for date in args.dates:
            print('\n' + '=' * 78)
            print('{}  {}'.format(config.name, date))
            print('=' * 78)
            try:
                if args.skip_transfer:
                    date_dir = config.targetDir(date)
                    print('  transfer skipped; using {}'.format(date_dir))
                else:
                    date_dir = gatherDate(config, date, dry_run=args.dry_run,
                                          interactive=not args.non_interactive)
                if args.transfer_only:
                    print('  transfer only; not attaching')
                    continue
                if args.dry_run:
                    print('  would attach {}'.format(date_dir))
                    continue
                report = attachDate(config, date_dir, attach_kwargs)
                quality = {p.get('quality') for p in (report or {}).get('plans', [])}
                print('{}: attached, quality={}'.format(date, quality or {'no-daq'}))
            except Exception as err:
                failures.append((date, '{}: {}'.format(type(err).__name__, err)))
                print('{}: FAILED -- {}: {}'.format(date, type(err).__name__, err))
    finally:
        # Do not leave a password sitting in memory once the run is over.
        clearCachedPasswords()

    print('\n' + '=' * 78)
    print('SUMMARY: {} / {} dates completed'.format(len(args.dates) - len(failures), len(args.dates)))
    for date, message in failures:
        print('  FAILED {}: {}'.format(date, message))
    print('=' * 78)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
