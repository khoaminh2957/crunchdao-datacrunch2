# CrunchDAO Hub API — Project Creation Research

## TL;DR
The Hub API has **no SDK endpoint for creating a new project**. Project creation is **web-only**. The CLI workflow is:
1. User clicks "Submit" on web UI -> Hub issues a `cloneToken`.
2. CLI calls `POST /v2/project-tokens/upgrade` with that `cloneToken` -> returns a permanent `pushToken` bound to the (already-created) project.
3. All subsequent CLI calls authenticate with that `pushToken`.

To create a new project, you must drive the web UI (or reverse the web's internal API, which is not exposed in the public SDK).

## Auth Schemes (from `crunch/api/_auth.py`)
| Scheme | Where | Header / Param |
|---|---|---|
| `ApiKeyAuth` | env `CRUNCHDAO_API_KEY` | `Authorization: API-Key <key>` |
| `PushTokenAuth` | `.crunchdao/token` | `?pushToken=<token>` (GET) or body field (POST) |
| `RunTokenAuth` | cloud runner | `X-Run-Token: <token>` |

The token in `.crunchdao/token` is a **pushToken**, scoped only to that one project. It returns 403 on any URL that is not `/v3|v4/competitions/datacrunch-2/projects/12867/curly-crayfishwetminh-v2/...`.

## POST Endpoints in SDK (grep `self.post`)
| Method | URL | Purpose |
|---|---|---|
| POST | `/v2/project-tokens/upgrade` | exchange `cloneToken` -> `pushToken` |
| POST | `/v3/competitions/{c}/projects/{u}/{p}/runs` (runner.py) | trigger runs |
| POST | `/v4/competitions/{c}/projects/{u}/{p}/submissions` | submit code |
| POST | `/v4/.../submissions/next-encryption-id` | get encryption id |
| POST | `/v3/.../submissions/{n}/files` | upload submission files |
| POST | upload-related (multipart) | file uploads |

**No POST on `/projects` itself.** Confirmed by grep over the entire `crunch` package.

## Curl Tests (token = pushToken for project 12867/curly-crayfishwetminh-v2)
```
GET /v3/competitions/datacrunch-2/projects/12867
  -> 403 AUTHORIZATION_DENIED  (pushToken too narrow)
GET /v3/competitions/datacrunch-2/projects/12867/curly-crayfishwetminh-v2
  -> 403 AUTHORIZATION_DENIED  (also denied via pushToken)
GET /v4/competitions/datacrunch-2/projects/12867/curly-crayfishwetminh-v2/clone?submissionNumber=1
  -> 200 {"main.py":"https://...s3...presigned"}   WORKS
GET /v3/users/@me                       -> 404 NO_RESOURCE_FOUND
GET /v3/competitions/datacrunch-2       -> 404 NO_RESOURCE_FOUND
```

`Authorization: API-Key <pushToken>` also returns 403 — pushTokens are not API-Keys.

## Curl Template (working call with pushToken)
```bash
TOKEN=$(cat .crunchdao/token)
curl -s -G --data-urlencode "pushToken=$TOKEN" \
  "https://api.hub.crunchdao.com/v4/competitions/datacrunch-2/projects/12867/curly-crayfishwetminh-v2/clone"
```

## How to Create a New Project (only known path)
1. Open `https://hub.crunchdao.com/competitions/datacrunch/submit` in a logged-in browser session.
2. Web UI shows a `crunch setup datacrunch-2 <new-project-name> --token <cloneToken>` command (already automated in `fetch_setup_token.py`).
3. Run that command -> CLI calls `POST /v2/project-tokens/upgrade` -> writes `.crunchdao/{project.json, token}`.

To fully automate without the browser you would need either a real `ApiKeyAuth` issued for user 12867 (request from CrunchDAO) or to reverse the web app's session-cookie endpoints — neither exists in the public SDK.
