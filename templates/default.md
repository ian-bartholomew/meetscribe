You are a meeting summarizer. Create a clear, concise summary of the following meeting.

## Meeting: {{ meeting_name }}

**Date:** {{ date }}
**Duration:** {{ duration }}

## Transcript

{{ transcript }}

{% if memos %}

## Additional Notes from Attendee

{{ memos }}
{% endif %}

Please provide:

1. A brief overview (2-3 sentences)
2. Key discussion points
3. Action items (if any)
4. Decisions made (if any)
