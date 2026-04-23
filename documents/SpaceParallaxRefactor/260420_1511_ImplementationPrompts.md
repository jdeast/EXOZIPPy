```markdown
# PHASE 1: SKETCH THE HIGHEST-LEVEL FLOW

## LLM INTERACTION GUIDELINES

When working through this phase with the user:

- Ask one question at a time
- Label possible answers A, B, C (or similar) for clarity
- Never assume details; ask for more information instead of guessing
- If a specification is ambiguous, ask for clarification before proceeding
- If you're about to make a significant decision, ask the user first
- Explicitly state what you don't understand or need more context for
- Wait for user confirmation before considering the phase complete
- Flag any contradictions or unclear requirements immediately

---

## CODE CONVENTIONS TO FOLLOW

When creating pseudocode and sketches, reference these conventions from the 
existing codebase:

**Docstrings:**
- Use NumPy format with Parameters, Returns, Raises sections
- Use `----------` underlines for section headers

**Naming:**
- Private methods: Prefix with `_`
- Config attributes: Listed in `CONFIG_KEYS` class variable

**State Management:**
- Use `_get_state()` and `_restore_state()` for pickling

**Output:**
- Use `self.output.log()` for messages
- Use `self.output.save_plot()` for figures

**Grid Searches:**
- Inherit from `BaseRectGridSearch`
- Implement `_setup_grid()` and `_fit_grid_point()`

**Floating Point:**
- All floating point grid coordinates should be rounded to 10 decimal places

---

## YOUR TASK

You are helping sketch the highest-level flow of a refactored workflow.

Starting from the public API entry point (or main workflow trigger), map out how the 
system should work at the highest level of abstraction.

**For the main entry method:**

1. Write pseudocode showing the major steps it must perform
2. Identify each step's responsibility (what does it accomplish?)
3. Sketch data flow between steps (what's passed from one to the next?)
4. Note where complexity lives (which steps will likely need sub-methods?)
5. List any assumptions you're making about how this should work

**Format your pseudocode like this:**

```
def main_method_name(input_data):
    """
    High-level description of what the workflow accomplishes.
    """
    # Step 1: Validate and normalize input
    #   - Takes: raw input_data
    #   - Produces: clean, validated data structure
    #   - Risk: edge cases in validation
    
    # Step 2: Do core transformation [COMPLEX - will likely need sub-methods]
    #   - Takes: validated data
    #   - Produces: transformed output
    #   - Risk: business logic is intricate, multiple scenarios
    
    # Step 3: Apply business rules
    #   - Takes: transformed data
    #   - Produces: final result
    #   - Risk: rules may conflict, need careful ordering
    
    # Step 4: Return result
    #   - Takes: final result
    #   - Produces: API response
```

Mark steps as [SIMPLE] or [COMPLEX] based on whether they'll need sub-methods.

---

## CONTEXT PROVIDED

### Specifications:
[User should paste their detailed specifications here]

### Existing Code:
[User should paste relevant source files here]

### Project Context:
[User should describe what this system does and why it's being refactored]

---

## OUTPUT

Provide:

1. **Main entry method pseudocode** (formatted as above)
2. **Data flow diagram** (text-based description of how data moves through steps)
3. **Complexity assessment** (which steps are [SIMPLE] and which are [COMPLEX]?)
4. **Assumptions** (what are you assuming about the spec or architecture?)
5. **Questions for clarification** (anything unclear that would change the flow?)

---

## NEXT STEPS

Once you've provided the above, I will review it with you and ask clarifying questions 
until we're confident this high-level flow is correct. Only then will we proceed to 
Phase 2.

**Do not proceed to the next phase until you explicitly confirm you're satisfied 
with this high-level flow.**
```

---

```markdown
# PHASE 2: COMPARE HIGH-LEVEL FLOW TO EXISTING CODE

## LLM INTERACTION GUIDELINES

When working through this phase with the user:

- Ask one question at a time
- Label possible answers A, B, C (or similar) for clarity
- Never assume details; ask for more information instead of guessing
- If something in the existing code is unclear, ask the user to explain it
- If you're unsure whether existing code can be reused, ask the user
- Wait for user confirmation before considering the phase complete
- Flag any concerns about integration or compatibility immediately

---

## CODE CONVENTIONS TO FOLLOW

When analyzing existing code and making recommendations, reference these conventions:

**Docstrings:**
- Use NumPy format with Parameters, Returns, Raises sections
- Use `----------` underlines for section headers

**Naming:**
- Private methods: Prefix with `_`
- Config attributes: Listed in `CONFIG_KEYS` class variable

**State Management:**
- Use `_get_state()` and `_restore_state()` for pickling

**Output:**
- Use `self.output.log()` for messages
- Use `self.output.save_plot()` for figures

**Grid Searches:**
- Inherit from `BaseRectGridSearch`
- Implement `_setup_grid()` and `_fit_grid_point()`

**Floating Point:**
- All floating point grid coordinates should be rounded to 10 decimal places

---

## YOUR TASK

Now that you have a high-level flow sketch, compare it to the existing codebase to 
identify what can be reused and what must be replaced.

**For the current entry point(s):**

1. Identify the existing main method and trace its flow
2. Map the new flow to the old flow
   - Which steps already exist and can be reused?
   - Which steps need to be replaced?
   - Which steps are entirely new?
3. Note the existing architecture
   - How is the current code organized?
   - What patterns/conventions does it follow?
   - Where are utilities/helpers that might support the new flow?
4. Identify integration constraints
   - What else depends on the main entry point?
   - Will the refactored code coexist with old code, or fully replace it?
   - Are there compatibility requirements?

---

## CONTEXT PROVIDED

### Phase 1 Output (High-Level Flow):
[User should paste Phase 1 pseudocode and analysis here]

### Existing Source Code:
[User should paste the current main entry point and related methods here]

### Existing Patterns/Utilities:
[User should paste any utility methods, helpers, or base classes that might be relevant]

### Project Context:
[Compatibility requirements, migration strategy, etc.]

---

## OUTPUT

Provide:

1. **Current main method analysis** (trace through existing code, how does it work?)
2. **Mapping of new flow to old flow** (step-by-step, what stays/changes/is new)
3. **Architectural patterns identified** (naming, structure, error handling, dependencies)
4. **Integration points** (where does refactored code connect to existing code?)
5. **Reusability assessment** (which existing methods can be reused as-is, which need modification, which must be replaced?)
6. **Integration constraints** (must coexist? Phased migration? Breaking changes?)
7. **Viability assessment** (given existing code and constraints, is the Phase 1 architecture still sound, or does it need revision?)

---

## CRITICAL QUESTION

Based on your analysis, should we proceed with the Phase 1 architecture, or does 
the existing code structure suggest revisions?

**Answer one of:**

A) Proceed as-is; Phase 1 architecture aligns well with existing code
B) Minor revisions needed to Phase 1; here's what should change...
C) Major rework needed to Phase 1; the existing code constrains us differently

---

## NEXT STEPS

Once you've provided the above output, we will review it together. If answer is B or C, 
we'll revise Phase 1 before proceeding. If answer is A, we'll move to Phase 3 
(Recursive Decomposition).

**Wait for explicit confirmation before proceeding.**
```

---

```markdown
# PHASE 3: RECURSIVE DECOMPOSITION (One Layer at a Time)

## LLM INTERACTION GUIDELINES

When working through this phase with the user:

- Ask one question at a time
- Label possible answers A, B, C (or similar) for clarity
- Never assume complexity levels; ask the user if unsure whether something needs further decomposition
- Ask before making significant structural decisions
- If decomposing a complex step, ask the user to validate it makes sense before continuing
- Explicitly state when you've completed one decomposition layer and ask before moving deeper
- Wait for confirmation before marking a step as "fully decomposed"

---

## CODE CONVENTIONS TO FOLLOW

When creating decomposition sketches and pseudocode, reference these conventions:

**Docstrings:**
- Use NumPy format with Parameters, Returns, Raises sections
- Use `----------` underlines for section headers

**Naming:**
- Private methods: Prefix with `_`
- Config attributes: Listed in `CONFIG_KEYS` class variable

**State Management:**
- Use `_get_state()` and `_restore_state()` for pickling

**Output:**
- Use `self.output.log()` for messages
- Use `self.output.save_plot()` for figures

**Grid Searches:**
- Inherit from `BaseRectGridSearch`
- Implement `_setup_grid()` and `_fit_grid_point()`

**Floating Point:**
- All floating point grid coordinates should be rounded to 10 decimal places

---

## YOUR TASK

Now take the high-level flow from Phase 1 and progressively decompose it, layer by layer.

**Decomposition Process:**

For each step in the current flow that is marked as [COMPLEX]:

1. Create a decomposition sketch (like Phase 1, but for this step)
   - Write pseudocode for how this step works internally
   - Identify its sub-steps
   - Show data flow between sub-steps
   - Mark which sub-steps are still [COMPLEX]

2. Compare to existing code (is there code that already does part of this?)

3. Review for soundness (does this decomposition make sense?)

4. Continue recursively (take remaining [COMPLEX] sub-steps and repeat)

**Continue until all steps are marked [SIMPLE]** and could be implemented in a 
single method without further breaking down.

---

## CONTEXT PROVIDED

### Phase 1 Output (High-Level Flow):
[User should paste Phase 1 pseudocode here]

### Phase 2 Output (Existing Code Analysis):
[User should paste Phase 2 mapping and analysis here]

### Existing Source Code (for reference):
[User should paste relevant existing methods that might guide decomposition]

---

## OUTPUT FORMAT

Output a hierarchical decomposition tree showing:

```
main_workflow(input)
├── Step 1: Validate [SIMPLE - implement directly]
├── Step 2: Transform [was COMPLEX, now decomposed]
│   ├── Sub-step 2.1: Parse [SIMPLE]
│   ├── Sub-step 2.2: Apply rules [COMPLEX - needs further decomposition]
│   │   ├── Sub-step 2.2.1: Load rules [SIMPLE]
│   │   ├── Sub-step 2.2.2: Execute rules [SIMPLE]
│   │   └── Sub-step 2.2.3: Merge [SIMPLE]
│   └── Sub-step 2.3: Validate [SIMPLE]
├── Step 3: Business rules [SIMPLE - implement directly]
└── Step 4: Return [SIMPLE]
```

For each decomposition layer, also provide:

1. **Pseudocode** for that layer's methods
2. **Data flow notes** (what passes between sub-steps)
3. **Reusability notes** (which existing methods can be reused here?)
4. **Soundness check** (does this decomposition make sense?)

---

## CRITICAL QUESTION

Is the decomposition complete? 

**Answer one of:**

A) All leaf-node steps are [SIMPLE] and fully decomposed
B) Some steps still marked [COMPLEX]; here's what still needs decomposition...
C) I'm unsure whether [specific step] is sufficiently decomposed; need guidance

---

## NEXT STEPS

Once you confirm answer A (all steps are [SIMPLE]), we will move to Phase 4 
(Architectural Review).

**Do not claim decomposition is complete until you're confident every leaf-node 
step could actually be implemented directly.**
```

---

```markdown
# PHASE 4: ARCHITECTURAL REVIEW (Multi-Level)

## LLM INTERACTION GUIDELINES

When working through this phase with the user:

- Ask one question at a time
- Label possible answers A, B, C for clarity
- Never assume something is fine; explicitly check each review criterion
- If you find an issue, describe it clearly and ask the user how to fix it
- If a decision is ambiguous, ask the user which approach they prefer
- Flag all issues before asking whether to proceed
- Ask the user to confirm each issue is resolved before moving on

---

## CODE CONVENTIONS TO FOLLOW

When reviewing the decomposition tree, reference these conventions:

**Docstrings:**
- Use NumPy format with Parameters, Returns, Raises sections
- Use `----------` underlines for section headers

**Naming:**
- Private methods: Prefix with `_`
- Config attributes: Listed in `CONFIG_KEYS` class variable

**State Management:**
- Use `_get_state()` and `_restore_state()` for pickling

**Output:**
- Use `self.output.log()` for messages
- Use `self.output.save_plot()` for figures

**Grid Searches:**
- Inherit from `BaseRectGridSearch`
- Implement `_setup_grid()` and `_fit_grid_point()`

**Floating Point:**
- All floating point grid coordinates should be rounded to 10 decimal places

---

## YOUR TASK

Review the complete decomposition tree for consistency, correctness, and completeness.

**Per-level review** (for each decomposition layer):

- Are the sub-steps in logical order?
- Is data flowing correctly from step to step?
- Are there any circular dependencies?
- Does each step have exactly one clear responsibility?
- Missing error handling or edge cases?

**Final global review** (after all layers checked):

1. **Check for missing dependencies**
   - Do all methods have access to data they need?
   - Are there shared utilities or state that should be extracted?
   - Any circular dependencies across layers?

2. **Check for pattern consistency**
   - Do all methods at the same level follow similar structures?
   - Are naming conventions consistent throughout the tree?
   - Do similar operations use similar patterns?

3. **Check for completeness**
   - Is every leaf-node method actually implementable as written?
   - Are all methods called somewhere in the tree?
   - Does the full tree accomplish the spec's requirements?

4. **Check integration points**
   - Where does the refactored code connect to existing code?
   - Are those connection points clean and minimal?
   - Will integration happen naturally as we implement top-down?

5. **Check error handling**
   - Where can errors occur at each level?
   - Are they handled at the appropriate level?
   - Will errors propagate correctly up the tree?

---

## CONTEXT PROVIDED

### Phase 3 Output (Decomposition Tree):
[User should paste the complete decomposition tree here]

### Phase 2 Output (Existing Code Analysis):
[User should paste Phase 2 mapping for reference]

---

## OUTPUT

Provide a review organized by category:

1. **Per-level review findings** (any issues at individual decomposition layers?)
2. **Missing dependencies** (issues with data flow or shared state?)
3. **Pattern consistency** (naming, structure, or convention issues?)
4. **Completeness** (missing methods, orphaned methods, or unmet requirements?)
5. **Integration points** (are connection points to existing code clean?)
6. **Error handling** (is error handling at the right levels?)
7. **Overall assessment** (is the architecture sound, or are there issues to fix?)

---

## CRITICAL QUESTION

Is the decomposition tree ready to move forward to testing and implementation?

**Answer one of:**

A) Architecture is sound; no changes needed. Ready to proceed to Phase 5 (Testing).
B) Issues found; here's what needs to be revised before proceeding...
C) Major architectural problems; we need to go back and revise Phase 1 or 3.

---

## NEXT STEPS

If answer is A: We're ready for Phase 5 (Create Unit Tests).

If answer is B: We'll revise the decomposition tree based on your findings, then 
review again.

If answer is C: We'll discuss which phases need revision and start over on those.

**Wait for explicit confirmation of which answer, and any needed revisions, 
before proceeding.**
```

---

```markdown
# PHASE 5: CREATE UNIT TESTS (Top-Down)

## LLM INTERACTION GUIDELINES

When working through this phase with the user:

- Ask one question at a time
- Label possible answers A, B, C for clarity
- Ask the user what test framework they prefer (pytest, unittest, etc.)
- Ask about any special test fixtures or helpers they use
- If you're unsure what valid test data looks like, ask the user
- Confirm the integration test covers the right end-to-end scenario
- Ask whether tests should use real or mocked dependencies at each layer
- Wait for confirmation before considering tests complete

---

## CODE CONVENTIONS TO FOLLOW

When writing tests, reference these conventions and follow the existing codebase's 
testing patterns:

**Docstrings:**
- Use NumPy format with Parameters, Returns, Raises sections
- Use `----------` underlines for section headers

**Naming:**
- Private methods: Prefix with `_`
- Config attributes: Listed in `CONFIG_KEYS` class variable

**State Management:**
- Use `_get_state()` and `_restore_state()` for pickling

**Output:**
- Use `self.output.log()` for messages
- Use `self.output.save_plot()` for figures

**Grid Searches:**
- Inherit from `BaseRectGridSearch`
- Implement `_setup_grid()` and `_fit_grid_point()`

**Floating Point:**
- All floating point grid coordinates should be rounded to 10 decimal places

---

## YOUR TASK

Before writing any implementation, write the tests. Test from the top down.

**Step 1: Integration test for main workflow**

Write a single end-to-end test for the main entry point:
- Use realistic input that exercises the full flow
- Verify the final output is correct
- This is your "proof that the architecture works" test

Example:
```python
def test_main_workflow_happy_path():
    """Main workflow processes valid input end-to-end."""
    input_data = {...realistic data...}
    result = main_workflow(input_data)
    assert result == {...expected output...}
```

**Step 2: Layer-by-layer unit tests**

For each layer of decomposition, create unit tests for those methods:

For **each method** at this layer:
- **Happy path test**: Normal usage with valid inputs
- **Edge case tests**: Boundary conditions, empty inputs, etc.
- **Error case tests**: Invalid inputs, exceptions raised, etc.
- **Isolation**: Mock any methods from deeper layers (not implemented yet)

Example test structure:
```python
def test_do_core_transformation_simple_data():
    """Transform produces expected output for simple input."""
    input_data = {...}
    result = do_core_transformation(input_data)
    assert result == {...expected...}

def test_do_core_transformation_empty_data():
    """Transform handles empty input gracefully."""
    result = do_core_transformation([])
    assert result == []

def test_do_core_transformation_invalid_input_raises_error():
    """Transform raises ValueError for invalid input."""
    with pytest.raises(ValueError):
        do_core_transformation(None)
```

---

## CONTEXT PROVIDED

### Phase 3 Output (Decomposition Tree):
[User should paste the complete decomposition tree here]

### Phase 4 Output (Architectural Review):
[User should paste the reviewed and approved architecture here]

### Specifications:
[User should paste relevant parts of the spec that define expected behavior]

### Existing Test Files (as reference):
[User should paste examples of existing tests to match style and conventions]

---

## QUESTIONS BEFORE I WRITE TESTS

Before creating the test suite, I need to ask:

**Question 1: Test Framework**
Which testing framework should I use?
A) pytest
B) unittest
C) Something else (please specify)

**Question 2: Test Data**
For the integration test, what should realistic input data look like? 
Please provide an example of valid input for the main workflow.

**Question 3: Expected Output**
For the integration test, what should the expected output look like?
Please provide an example of what successful completion should produce.

**Question 4: Mocking Strategy**
When testing methods at layer N, should methods at layer N+1 be:
A) Mocked (we assume deeper layers work correctly)
B) Real implementations (test the full call stack)
C) Mixed (some mocked, some real - depends on the method)

**Question 5: Existing Test Utilities**
Are there existing fixtures, helpers, or mock objects I should use?
If so, please show me examples.

---

## OUTPUT

Once I receive answers to the above questions, I will provide:

1. **Integration test** (end-to-end test for main_workflow)
2. **Layer-by-layer unit tests** organized by decomposition layer
3. **Complete test file(s)** in executable form
4. **Test data/fixtures** needed to run the tests
5. **Comments** explaining the testing strategy for each layer

---

## NEXT STEPS

Once you provide answers to my questions and review the generated tests, we'll move 
to Phase 6 (Implementation).

**Wait for explicit confirmation that tests look correct before proceeding 
to implementation.**
```

---

```markdown
# PHASE 6: IMPLEMENT METHODS (Top-Down)

## LLM INTERACTION GUIDELINES

When working through this phase with the user:

- Ask one question at a time
- Label possible answers A, B, C for clarity
- Ask the user to confirm the implementation strategy makes sense before starting
- If you're implementing a method and need to call a stub, ask how deep to go
- Ask the user after each implemented method whether they want to see test results
- If tests fail, ask the user what should change (implementation or test?)
- Flag any issues with integration to existing code
- Ask for confirmation that implementation is complete before wrapping up

---

## CODE CONVENTIONS TO FOLLOW

When implementing methods, strictly follow these conventions:

**Docstrings:**
- Use NumPy format with Parameters, Returns, Raises sections
- Use `----------` underlines for section headers

**Naming:**
- Private methods: Prefix with `_`
- Config attributes: Listed in `CONFIG_KEYS` class variable

**State Management:**
- Use `_get_state()` and `_restore_state()` for pickling

**Output:**
- Use `self.output.log()` for messages
- Use `self.output.save_plot()` for figures

**Grid Searches:**
- Inherit from `BaseRectGridSearch`
- Implement `_setup_grid()` and `_fit_grid_point()`

**Floating Point:**
- All floating point grid coordinates should be rounded to 10 decimal places

---

## YOUR TASK

Now implement the actual code, working from the top layer downward.

**Implementation Strategy:**

1. **Start with main entry point**
   - Replace pseudocode with real logic
   - Call sub-methods (even if they don't exist yet—use placeholders/stubs)
   - Implementation structure should match decomposition tree exactly

2. **Then implement one layer at a time**
   - Pick one method from the next decomposition layer
   - Implement it fully (replacing its pseudocode)
   - Call its sub-methods (which can be stubs)
   - Run its unit tests (from Phase 5)
   - They should pass

3. **Continue recursively downward**
   - For each method you implement, replace the stubs it calls with real implementations
   - Run tests after each method is complete
   - Move to the next method at the same layer, then go deeper

4. **Handle integration**
   - Integrate reused methods from Phase 2 properly
   - Remove old code that's been replaced
   - Update any dependent code that calls these methods

5. **Final verification**
   - Run the full test suite
   - Run the end-to-end integration test from Phase 5
   - Verify the refactored code works with the rest of the system

---

## CONTEXT PROVIDED

### Phase 5 Output (Test Suite):
[User should paste the complete test file(s) here]

### Phase 3 Output (Decomposition Tree):
[User should paste the complete decomposition tree here]

### Phase 2 Output (Reusability Analysis):
[User should paste which methods can be reused from existing code]

### Existing Source Code:
[User should paste the existing code that will be refactored/integrated]

---

## QUESTIONS BEFORE I IMPLEMENT

**Question 1: Implementation Order**
Should I implement all methods at layer 1, then all methods at layer 2?
Or should I go deep (implement main_workflow + all its dependencies + all their dependencies)?

A) Breadth-first (all layer 1, then all layer 2, etc.)
B) Depth-first (main_workflow and its full call stack)
C) By functional area (do all transformation-related methods, then all validation methods)

**Question 2: Stub Pattern**
When a method calls a sub-method that hasn't been implemented yet, should stubs:
A) Return None with a comment
B) Raise NotImplementedError()
C) Return a dummy/mock value appropriate to that method's return type
D) Something else?

**Question 3: Reused Methods**
I identified these methods from Phase 2 that can be reused:
[List them here]

Should I:
A) Use them as-is without modification
B) Review and modify them if needed to fit new flow
C) Ask before modifying any existing method?

**Question 4: Integration Approach**
When removing old code and integrating new code:
A) Remove old code completely as we implement new code
B) Keep old code during implementation, remove it at the end
C) Gradually transition (support both old and new during refactor)?

---

## OUTPUT

Once I receive answers to the above questions, I will provide:

1. **Main entry point implementation** (with stubs for deeper layers)
2. **Layer-by-layer implementation** (in implementation order)
3. **Integration notes** for each layer (what existing code changed, how it connects)
4. **Final wiring and cleanup** (removing old code, ensuring everything connects)
5. **Test results** (which tests pass/fail after implementation)

Each implementation will:
- Follow the decomposition tree structure
- Use the pseudocode from Phase 3 as the guide
- Follow all code conventions specified above
- Include complete docstrings
- Reference existing patterns from the codebase

---

## NEXT STEPS

Once you answer my questions, I'll implement the code layer by layer. After each 
layer, I'll show you the code and ask if you want to continue to the next layer or 
make adjustments.

**We'll iterate until everything is implemented and all tests pass.**
```

These 6 prompts are now **completely standalone**. Each one:
- Includes the interaction guidelines
- Includes the code conventions
- Provides context about what came before
- Asks clarifying questions before proceeding
- Has a clear signal for when it's done and waiting for review

You can now work through them one at a time, reviewing and making decisions between phases.