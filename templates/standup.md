Summarize the following meeting transcript as a standup summary.

## Meeting: {{ meeting_name }}

**Date:** {{ date }}

## Format

- **Yesterday:** What was discussed about past work
- **Today:** What was planned
- **Blockers:** Any blockers mentioned

## Transcript

{{ transcript }}

{% if memos %}

## Additional Notes

{{ memos }}
{% endif %}

Provide any action items.
