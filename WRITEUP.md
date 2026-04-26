# Ask First — AI Intern Assignment Writeup

## 1. How I Approached the Reasoning Problem

The biggest challenge in this assignment was figuring out how to teach the AI the difference between a real health pattern and just a random coincidence. To do this, I built a system that looks at the big picture rather than just matching keywords. Here’s how I tackled it:

**Reading the Whole Story at Once (No Chunking)**
Usually, when building AI apps, people use techniques like RAG or "sliding windows" to chop text into smaller chunks. I realized early on that this wouldn't work for health data. For example, hair loss (Telogen Effluvium) usually happens 8 to 12 weeks after a big change in diet or stress. If the AI is only looking at a small chunk of recent conversations, it completely misses that 3-month gap. So, I set it up to serialize and read the patient's *entire* history from start to finish in one go. This way, it can actually understand the passage of time.

**Forcing the AI to "Show Its Work"**
To make sure the AI wasn't just blindly associating words, I used a framework called **LangGraph** to separate the regular chat from the heavy pattern-finding work. For the pattern finding, I used **Pydantic** to force the LLM to output its answers in strict JSON. More importantly, I forced the AI to write down a "reasoning trace" (a step-by-step thought process) *before* giving the final answer. It has to point out exact session numbers, calculate the time lag in weeks, and explain the medical logic before drawing a conclusion.

**How the System Grades Confidence**
When the system spots a potential pattern, it needs to validate it. It asks itself a few questions:
- **Did the cause actually happen before the symptom?** (Basic, but important).
- **Does it happen repeatedly?** (If a trigger and symptom happen together 3 or 4 times, it's a stronger signal).
- **Did changing the habit fix the problem?** (If the user stopped eating late and their stomach pain went away, that’s a huge clue. The system gives these "natural experiments" the highest confidence scores).

## 2. Honest Failure Analysis & Future Improvements

### Where the AI gets confused (Failure Modes):
- **Blaming the wrong thing:** Sometimes a user will mention two things at once, like "I've been super stressed at work and I'm sleeping terribly." If they get a headache later, the AI sometimes just blames whatever was mentioned first in the sentence, rather than figuring out which one is the real cause.
- **Jumping to conclusions (Confirmation Bias):** The LLM has a bad habit of deciding on a theory during the first couple of sessions. Once it decides "coffee causes the headaches," it tends to ignore later sessions that might prove that theory wrong.
- **Is it just a coincidence?** With only about 10 sessions per user, it's hard to be mathematically sure about a pattern. The AI might give a "Very High" confidence score because two things happened together 3 times, but statistically, that could still just be random chance.

### What I would build if I had more time:

- **Reminders/Update seeking feature about health status:** Getting the update of the patients if they are doing well, Update or progress regarding their well-being so that it satisfies the user and give a feeling that the agent really cares about the patient's well-being and help him accordingly.

- **A "Devil's Advocate" Agent (Pattern Critic Node):** I would add a second AI agent whose only job is to try and prove the first AI wrong. It would actively search the history for times when the trigger happened but the symptom *didn't*.
- **Pre-counting Connections:** Before even asking the LLM to think, I'd write a script to simply count how often certain tags (like `late eating` and `stomach pain`) appear near each other. Giving the AI raw numbers helps it make better, data-driven decisions.
- **Actual Medical Knowledge (RAG):** Right now, the AI uses its general training to explain *why* a symptom happens. Sometimes it sounds super confident but gets the biology slightly wrong. I'd love to connect it to a real medical database (like PubMed) so its explanations are backed by real science.
- **Long-term Memory Summaries:** Reading the whole history works great for 10 sessions, but if a user has 100+ sessions, the AI will get overwhelmed. For long-term users, I'd build a system that summarizes older, boring sessions but keeps detailed records of major health events.


---
*Submission by: Danish Adnaan *
