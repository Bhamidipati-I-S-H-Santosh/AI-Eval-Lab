# Identity-aware access-control testing

Enterprise agents answer from permission-aware data: SharePoint, Google
Drive, Confluence, ticketing systems. A dangerous failure is an agent that
answers Bob with a document only Alice may read. evalkit tests for it by
asking the same questions as different test users and checking every piece
of evidence against the source's real permissions.

## How a violation is detected

For each answer, evalkit collects the documents the response exposed, from
strongest evidence to weakest:

| Evidence | Where it comes from |
|---|---|
| `retrieved` | documents the system returned as retrieved / cited (adapter or trace) |
| `tool_result` | document names inside tool results (e.g. a search tool's JSON) |
| `citation` | sources cited in the answer: `[file.md]`, markdown links |
| `content` | the facts a case expects (`expected_contains`) appear in the answer, and **every** expected source is denied to this user: a leak even without any citation |

Each document is checked with the ACL source: allowed, denied, or unknown.
`access_violation` is 1 for a run if anything denied was exposed
(`identity.unknown: violation` also counts unknown ACLs). Reports gain an
"Access control" section and `access_report.json` lists every violation
(question, identity, document, evidence) and a per-question comparison across
identities.

## Configuration

```yaml
identity:
  acl: {type: static, path: acl.yaml}
  users:
    alice: {email: alice@corp.example, groups: [hr]}
    bob:   {email: bob@corp.example, groups: [engineering]}
  matrix: true            # ask every case as every user
  unknown: report         # or: violation
gates:
  access_violation: "<=0"
```

### Making the system act as each user

Pick one:

- **Per-user credentials** (recommended): each user's `system` block is
  deep-merged into the main system config, e.g. a delegated token obtained for
  a test account:
  ```yaml
  users:
    bob: {email: bob@corp.example,
          system: {auth: {type: bearer, token: "${secret:bob-access-token}"}}}
  ```
- **Identity hook** (Python agents): `system: {type: python, target: ..., identity_hook: my_agent:act_as}`.
  Calls through a shared adapter are serialized so identities never mix.

Use dedicated test accounts with realistic group memberships in a test tenant
or with read access to a test site.

## ACL sources

| `type` | Document reference | Permissions read from |
|---|---|---|
| `static` | doc id / `metadata.source`, globs | YAML: `{"salary_bands.md": {allow: [group:hr], deny: [...]}}`; a deny anywhere wins, otherwise the most specific pattern decides |
| `sharepoint` | `metadata.drive_id` + `item_id`, or a SharePoint URL (`web_url`/`url`) | Microsoft Graph item permissions (user, group, site group, organization/anonymous links); group membership via `transitiveMemberOf` |
| `google_drive` | `metadata.file_id` or a Drive URL | Drive API v3 permissions (user, group, domain, anyone) |
| `confluence` | `metadata.page_id` or a `/pages/<id>/` URL | page read restrictions including all ancestors |

`doc_map: {evalkit-doc-key: provider-ref}` maps ids when the system's document
ids don't carry provider references. Custom sources implement
`evalkit.identity.AclSource.allowed(user, doc)` ([plugins.md](plugins.md)).

Permissions needed (read-only): Graph `Sites.Read.All` + `GroupMember.Read.All`;
Drive `drive.metadata.readonly`; Confluence read access to the spaces.

## Limits

- Detection relies on the evidence the system exposes. An agent that
  paraphrases a restricted document without citing it is caught only through
  `content` checks, so write cases whose `expected_contains` facts exist only
  in restricted documents.
- Confluence space-level permissions are not evaluated (pages without
  restrictions are treated as readable; set `unrestricted: unknown` to flag them).
- SharePoint sharing links scoped to specific people are matched by the
  identities Graph returns for them.
