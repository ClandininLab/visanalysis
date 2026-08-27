# Config format

One JSON (or YAML, if PyYAML is installed) file per user. Keys:

| key | meaning |
|---|---|
| `name` | label for logs; defaults to the filename |
| `plugin` | `fortyhourfitness`, `twentyfourhourfitness`, `bruker`, `aodscope` |
| `target.path` | where date directories are assembled and attached |
| `target.date_format` | `strftime` spelling of the date directory, default `%Y-%m-%d` |
| `target.permissions` | octal string, e.g. `"774"`; omit to leave permissions alone |
| `sources[]` | one entry per place data is pulled from, copied in order |
| `sources[].type` | `local` (a path or mount) or `sftp` |
| `sources[].path` | the directory *containing* the date directories |
| `sources[].date_format` | that source's own date spelling - they often differ |
| `sources[].required` | `false` lets a missing source warn instead of fail (default `true`) |
| `sources[].host` / `port` / `username` | SFTP only |
| `sources[].auth.method` | `prompt` (default), `env`, `key`, or `agent` |
| `attach` | defaults forwarded to the plugin's `attachData`; CLI flags override |

Paths may use `~` and `$VARS`.

## Date formats differ between sources

This is the most common gotcha. On the 40-hour rig the Jackfish/assist machine writes
`20260819` while the rig machine writes `2026-08-19`, for the same day. Set `date_format`
per source and the script reconciles them into one target directory.

If a `date_format` cannot be applied (the argument is not `YYYY-MM-DD`), the argument is used
verbatim, so directories that are not date-shaped still work.

## Credentials

**Never put a password in a config file.** These files live in the repo and get committed.
A config containing `password`, `secret`, `token`, `passphrase`, or similar is rejected
outright rather than warned about.

Pick one:

- `{"method": "agent"}` — SSH agent or a default key. Best option; nothing to type or store.
- `{"method": "key", "key_filename": "~/.ssh/id_rsa"}` — a specific key file.
- `{"method": "env", "var": "RIG_SFTP_PASSWORD"}` — for cron/CI, where nobody can type.
- `{"method": "prompt"}` — default. Asks once per host per run via `getpass`, keeps it in
  memory only, and clears it when the run ends.

## Where configs are found

`--config <name>` looks for `<name>.{json,yaml,yml}` in, in order:

1. the path as given, if it is a file
2. `rig_specific/configs/`
3. `~/.config/visanalysis/`

With no `--config`: `$VISANALYSIS_RIG_CONFIG`, then `~/.config/visanalysis/rig_config.json`.

Keep a personal config out of git by putting it in `~/.config/visanalysis/` rather than in
`configs/`. Files starting with `_` are not listed by `--list-configs`.
