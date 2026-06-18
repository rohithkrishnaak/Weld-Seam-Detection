# Git Commit Formatting Protocol

You are an automated version control assistant. When tasked with creating a commit message, you must evaluate the recent changes and format the message according to this strict structure:

1. **Phase Tag:** Start the commit message with the relevant project phase:
   - [Phase 1: Calibration]
   - [Phase 2: DL ROI Model]
   - [Phase 3: Fuzzy Pipeline]
   - [Phase 4: Extraction Geometry]
   - [Phase 5: Integration]
   - [Phase 6: Evaluation]
2. **Action Category:** Follow the phase tag with one of: (feat, fix, refactor, docs, chore).
3. **Description:** Provide a concise, imperative-tense description of the change.
4. **Details:** If the commit involves mathematical logic (e.g., TFN boundaries, Steger's Hessian), list the specific parameters modified.

Example: `[Phase 3: Fuzzy Pipeline] feat: implement vectorized FCM clustering with multi-dimensional feature vectors`