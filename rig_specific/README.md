# rig_specific

Rig- and user-specific entry points. One script, many users: everything that differs between
people lives in a small config file rather than in a forked copy of the script.

## move_and_attach_data.py

Copies a day's acquisition data from its sources into one target date directory, then attaches
it to the stimpack hdf5 with the appropriate plugin.

```bash
# see what configs exist
python move_and_attach_data.py --list-configs

# check a config resolves the way you expect, without touching anything
python move_and_attach_data.py --config my_config --show-config
python move_and_attach_data.py --config my_config --dry-run 2026-08-19

# do it
python move_and_attach_data.py --config my_config 2026-08-19 2026-08-20
```

Exits non-zero if any date failed, and prints a summary naming which. A failure on one date no
longer abandons the rest.

### Options

| flag | effect |
|---|---|
| `--config NAME\|PATH` | which config to use |
| `--list-configs`, `--show-config` | inspect without running |
| `--dry-run` | print the plan, change nothing |
| `--skip-transfer` / `--transfer-only` | run only half the pipeline |
| `--non-interactive` | never prompt; auth must be `env`, `key`, or `agent` |
| `--no-ensemble` | never split a `.jfdaqdata`; one block per file |
| `--require-ensemble` | fail if any `.jfdaqdata` yields a single block |
| `--counts-only` | allow a cut from FicTrac line counts alone, when no camera stopped |
| `--force` | downgrade failing validation checks to warnings |
| `--no-verify` | skip the photodiode audit |
| `--ensemble-json PATH` | manual segmentation override |

Attach flags are forwarded only to plugins that accept them, so a config written for the
40-hour rig does not break when pointed at `bruker` or `aodscope`.

## Configs

See [`configs/_README_config_format.md`](configs/_README_config_format.md) for the full schema.
`configs/example.json` is a template; `configs/fortyhr_michelle.json` is a working example.

Two things worth knowing up front:

**Sources may spell dates differently.** On the 40-hour rig the Jackfish/assist machine writes
`20260819` while the rig machine writes `2026-08-19` for the same day. Each source carries its
own `date_format`, and they are reconciled into one target directory.

**Passwords never go in a config.** A config containing `password`, `secret`, `token`, or
`passphrase` is rejected outright — these files sit in the repo and get committed. Use an SSH
key or agent where you can; otherwise the script prompts once per host per run via `getpass`,
holds it in memory only, and clears it when the run ends.

Keep a personal config in `~/.config/visanalysis/` rather than in `configs/` to stay out of git
entirely.

## Adding yourself

1. Copy `configs/example.json` to `~/.config/visanalysis/rig_config.json` (or to
   `configs/<yourname>.json` if it holds nothing private).
2. Set `target.path` and one entry in `sources` per machine you pull from.
3. Set each source's `date_format` to match how that machine names its date directories.
4. Check it with `--show-config`, then `--dry-run`, then run it.

No code change is needed to onboard a new user or a new rig layout.
