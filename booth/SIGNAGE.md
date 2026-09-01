# Booth signage (T11, spec section 8)

Print large, place at eye level next to the robot. Two variants — post
the one matching the mode actually running (the mode disclosure is a
spec requirement).

---

## Sign 1 — main disclosure (always up)

> ### This robot recognizes faces 🤖
>
> **Reachy the Language Tutor** remembers the people it teaches.
>
> - It only remembers you if you **say yes** when it asks.
> - What it stores: your first name, the language you're practicing,
>   its lesson notes, and a **numeric face signature** — *not* a photo,
>   and it can't be turned back into your picture.
> - Everything is stored **on the computer at this booth** and every
>   visitor profile is **deleted at the end of the day**.
> - Want out sooner? Just tell the robot **"forget me"** — it deletes
>   your file on the spot.
> - Kids are welcome to try it with a parent's okay.

---

## Sign 2a — local speech mode (default)

> **How it hears you:** all speech recognition and the robot's voice run
> on the computer at this booth. Only the *text* of the conversation is
> sent to Claude (Anthropic) to think up replies. No audio leaves the
> booth.

## Sign 2b — cloud speech mode (the mixed-language showpiece)

> **How it hears you:** in this demo the conversation audio streams to
> Google's Gemini Live API, which does the listening, thinking, and
> talking. Face data still never leaves the booth.

---

*Operator note: `./start_booth.sh` runs local mode; if the bake-off
switches a demo to `--speech cloud`, swap sign 2a for 2b before the
first visitor.*
