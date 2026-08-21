# Security and sensitive-data reporting

Please do not open a public issue for leaked credentials, participant-level
data, re-identification risks, or an exploitable vulnerability.

Use GitHub's private vulnerability reporting feature when it is available. If
it is unavailable, contact the corresponding author through the address given
in the accepted paper and include only the minimum information needed to
reproduce the problem. Do not send raw EEG, participant identifiers, API keys,
or access tokens by ordinary issue comments.

When reporting, include:

- the affected commit or release;
- the affected file/component;
- impact and minimal reproduction steps;
- whether credentials or participant data may already have been exposed.

Immediately revoke any exposed credential. Deleting it from the newest commit
does not remove it from Git history.
