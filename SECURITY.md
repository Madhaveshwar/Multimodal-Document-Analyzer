# Security Policy

## Supported Versions

Security fixes are applied to the current main branch and the latest public release.

## Reporting a Vulnerability

If you believe you have found a security issue, please report it privately rather than opening a public issue.

Include:

- A clear summary of the issue
- Steps to reproduce it
- Any affected files, endpoints, or workflows
- The potential impact

## Security Practices

- Store API keys and secrets in environment variables only.
- Do not commit `.env` files, database files, or log files.
- Sanitize uploaded filenames and user-controlled text before reuse.
- Rotate production secrets if they may have been exposed.
- Prefer least-privilege access for deployment accounts and cloud services.

## Deployment Notes

The application is designed to run behind a trusted reverse proxy or platform-managed ingress in production. Expose only the intended public ports and keep the health endpoint limited to operational status checks.