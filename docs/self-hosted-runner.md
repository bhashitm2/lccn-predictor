# Running the crawler on a dedicated IP

## Why

The crawler needs an IP that LeetCode's Cloudflare edge will serve. On a
GitHub-hosted runner that is a coin flip: runners draw from Azure ranges shared
with every CI user on the internet, and observed runs have been roughly 50/50
regardless of how politely we crawl. A refused run is refused on the *first*
request — before any pacing, backoff or retry logic can help:

```
non-200 403 for https://leetcode.com/contest/api/ranking/weekly-contest-517/ (attempt 1/6)
... 6 attempts, all 403 ...
CrawlBlockedError: ... this IP is being refused
```

Two hours earlier, the same code on a different runner pulled all 39,401 rows
with two transient 403s. The variable is the IP, not the crawler.

A ~$4–6/month VPS with a static IP removes that variable. Oracle Cloud's
always-free ARM tier also works if you can get capacity.

> **Before you spend anything:** GitHub Actions is free and unlimited on this
> public repo, and a fresh runner IP often works fine. This is worth doing if
> scheduled crawls fail repeatedly — not after one bad day.

## Security: why only *some* workflows may use it

This repository is **public**. Anyone can fork it and open a pull request. If a
workflow that a fork can trigger runs on your self-hosted runner, that PR
executes arbitrary code on your machine. This is a real, routinely exploited
attack — not a theoretical one.

| workflow | triggers | fork-triggerable? | runner |
|---|---|---|---|
| `crawl-cron.yml` | `schedule`, `workflow_dispatch` | no | self-hosted OK |
| `backfill.yml` | `workflow_dispatch` | no | self-hosted OK |
| `ci.yml` | `push`, `pull_request` | **yes** | `ubuntu-latest` **only** |

`ci.yml` hardcodes `runs-on: ubuntu-latest` with a comment explaining why.
**Never** point it at `vars.CRAWLER_RUNNER`.

Treat the runner as untrusted-adjacent regardless: dedicated non-root user, no
other services, nothing else of value on the box.

## Setup

Ubuntu 24.04, smallest tier (1 vCPU / 1–2 GB). The job is ~1,600 paced HTTPS
requests plus one FFT — neither CPU- nor memory-hungry.

### 1. Python

`requirements.txt` pins numpy 2.5.0, which needs Python ≥ 3.12. Ubuntu 24.04
ships 3.12, which works; 3.13 matches CI. Install it explicitly, because
`actions/setup-python` on a self-hosted runner cannot always fetch a version
the machine does not already have:

```bash
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.13 python3.13-venv python3.13-dev git curl
```

### 2. Runner, as a non-root user

```bash
sudo adduser --disabled-password --gecos "" runner
sudo -iu runner
mkdir actions-runner && cd actions-runner
# Take the current URL + token from:
#   Settings → Actions → Runners → New self-hosted runner
curl -o actions-runner-linux-x64.tar.gz -L <url-from-that-page>
tar xzf actions-runner-linux-x64.tar.gz
./config.sh --url https://github.com/bhashitm2/lccn-predictor \
            --token <token-from-that-page> \
            --labels crawler \
            --unattended
```

The `crawler` label is what `CRAWLER_RUNNER` will point at. Then install it as a
service so it survives reboot:

```bash
exit                       # back to your sudo user
cd /home/runner/actions-runner
sudo ./svc.sh install runner
sudo ./svc.sh start
sudo ./svc.sh status
```

### 3. Cut over

In **Settings → Secrets and variables → Actions → Variables**:

| variable | value | effect |
|---|---|---|
| `CRAWLER_RUNNER` | `crawler` | crawl + backfill run on your VPS |
| `CRAWLER_RATE_LIMIT` | `8` | optional; full crawl ~9 min → ~3.5 min |

No code change and no PR. **Deleting `CRAWLER_RUNNER` sends everything straight
back to GitHub-hosted runners** — that is your rollback.

### 4. Lock down Atlas (worth doing while you are here)

Atlas Network Access is almost certainly `0.0.0.0/0` today, because
GitHub-hosted runner IPs are unpredictable. With a fixed VPS IP you can narrow
it to that single address — a genuine security improvement you get for free by
moving. Remember the Render web host also connects, so allowlist both.

### 5. Verify

Dispatch **Crawl & predict latest contest** with slug `weekly-contest-517`,
force `true`. Expect `fetched ~39401 ranking rows` and
`status=done records=~39401`.

This is also how `weekly-contest-517` gets repaired. No scheduled run will ever
do it: crons call `predict-latest`, which targets the newest finished contest.

## Operating notes

- **Runner offline → jobs queue, they do not fail.** They sit until the
  110-minute timeout. If crawls stop appearing, check `sudo ./svc.sh status`
  before debugging anything else.
- **Disk**: the runner keeps workspaces and logs. Trim `_diag/` occasionally.
- **Updates**: the runner self-updates; the OS does not. Patch it.
- **The politeness settings still matter.** A dedicated IP is only clean while
  you treat it well — a static IP you hammer is a static IP you lose, and
  unlike a runner IP you cannot roll the dice on a new one.
