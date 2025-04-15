import streamlit as st
import time
from sheets import append_data_to_sheet
import json
import logging

def handle_user_input():
    if st.session_state.reset_input_box:
        st.session_state.input_box = ''
        st.session_state.reset_input_box = False

    # Render the input box outside the form
    main_input_box()

    with st.container():
        with st.form(key='problem_form'):
            submit_button = st.form_submit_button("Submit")

        if submit_button:
            st.session_state.user_input_submitted = True
            st.session_state.user_input = st.session_state.input_box
            # Update the input buffer when the form is submitted
            st.session_state.input_buffer = st.session_state.input_box

    if st.session_state.user_input_submitted:
        st.session_state.user_input_submitted = False
        user_input = st.session_state.input_box
        st.session_state.user_input = user_input

        if user_input:
            # Start new problem
            if st.session_state.problem_state['steps'] is None:
                st.session_state.problem_state['original_problem'] = user_input
                with st.spinner("Processing the problem, this could take a few minutes due to high traffic..."):
                    try:
                        solution = st.session_state.solver.get_math_solution(user_input)
                        st.session_state.problem_state['steps'] = solution.steps
                        st.session_state.problem_state['final_answer'] = solution.final_answer
                        st.session_state.problem_state['current_step'] = 0
                        st.session_state.problem_state['awaiting_answer'] = True
                        st.session_state.problem_state['solution'] = solution
                        st.session_state.problem_state['new_step'] = True

                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": f"Let's solve this problem step by step: {user_input}",
                            "timestamp": time.strftime("%H:%M"),
                            "requires_input": False
                        })

                        current_step = st.session_state.problem_state['current_step']
                        step = st.session_state.problem_state['steps'][current_step]
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": f"**Step {current_step +1}:** {step.instruction}\n\n{step.question}",
                            "timestamp": time.strftime("%H:%M"),
                            "requires_input": True,
                            "step_num": current_step
                        })
                        st.session_state.problem_state['new_step'] = False

                        st.session_state.user_input = ''
                        st.session_state.input_buffer = ''
                        st.session_state.reset_input_box = True
                        st.session_state.problem_state['awaiting_answer'] = True

                        data = {
                            "original_problem": user_input,
                            "current_step": 0,
                            "user_input": user_input,
                            "correct_answer": None,
                            "hint": None,
                            "final_answer": solution.final_answer,
                            "problem_summary": None,
                            "user_feedback": "Started new problem"
                        }
                        append_data_to_sheet(json.dumps(data))

                    except Exception as e:
                        print(e)
                        st.error(f"Error processing problem: {str(e)}")

            # Continue current problem
            else:
                current_step_index = st.session_state.problem_state['current_step']
                steps = st.session_state.problem_state['steps']
                current_step = steps[current_step_index]
                expected_answer = current_step.answer

                try:
                    # Add debug logging at the start
                    logging.debug("Starting handle_user_input")
                    
                    is_correct, explanation = st.session_state.solver.validate_step_answer_llm(user_input, expected_answer, current_step.question)
                    
                    # Dump the results to a file for inspection
                    st.session_state.solver.dump_to_file({
                        "is_correct": is_correct,
                        "explanation": explanation,
                        "user_answer": user_input,
                        "correct_answer": expected_answer,
                        "step_question": current_step.question
                    }, "debug_validation.txt")
                    
                    # Add debug logging
                    logging.debug(f"Validation result - is_correct: {is_correct}, explanation: {explanation}")
                    
                    if is_correct:
                        current_step.user_correct = True
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": f"✅ Correct! {explanation}",
                            "timestamp": time.strftime("%H:%M"),
                            "requires_input": False,
                            "step_num": current_step_index
                        })
                        st.session_state.problem_state['current_step'] += 1
                        st.session_state.user_input = ''
                        st.session_state.input_buffer = ''
                        st.session_state.reset_input_box = True

                        # Solved problem
                        if st.session_state.problem_state['current_step'] >= len(steps):
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": f"Great job! The final answer is: {st.session_state.problem_state['final_answer']}",
                                "timestamp": time.strftime("%H:%M"),
                                "requires_input": False
                            })

                            with st.spinner("Generating problem summary..."):
                                summary = st.session_state.solver.generate_problem_summary(
                                    st.session_state.problem_state['solution']
                                )
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": summary,
                                "timestamp": time.strftime("%H:%M"),
                                "requires_input": False
                            })

                            st.session_state.show_feedback_form = True

                            data = {
                                "original_problem": st.session_state.problem_state['original_problem'],
                                "current_step": "complete",
                                "user_input": user_input,
                                "correct_answer": expected_answer,
                                "hint": None,
                                "final_answer": st.session_state.problem_state['final_answer'],
                                "problem_summary": summary,
                                "user_feedback": "Problem completed"
                            }
                            append_data_to_sheet(json.dumps(data))

                        else:
                            next_step_num = st.session_state.problem_state['current_step']
                            next_step = st.session_state.problem_state['steps'][next_step_num]
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": f"**Step {next_step_num +1}:** {next_step.instruction}\n\n{next_step.question}",
                                "timestamp": time.strftime("%H:%M"),
                                "requires_input": True,
                                "step_num": next_step_num
                            })
                            st.session_state.problem_state['awaiting_answer'] = True

                        data = {
                            "original_problem": st.session_state.problem_state['original_problem'],
                            "current_step": current_step_index + 1,
                            "user_input": user_input,
                            "correct_answer": expected_answer,
                            "hint": current_step.explanation if current_step.hint_count > 0 else None,
                            "final_answer": st.session_state.problem_state['final_answer'],
                            "problem_summary": None,
                            "user_feedback": "Correct answer"
                        }
                        append_data_to_sheet(json.dumps(data))

                    else:
                        current_step.attempt_count += 1
                        remaining_attempts = 3 - current_step.attempt_count

                        if current_step.attempt_count >= 3:
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": f"❌ That's not quite right. {explanation}\nThe correct answer is: {expected_answer}",
                                "timestamp": time.strftime("%H:%M"),
                                "requires_input": False,
                                "step_num": current_step_index
                            })
                            current_step.user_correct = False

                            st.session_state.problem_state['current_step'] += 1
                            st.session_state.user_input = ''
                            st.session_state.input_buffer = ''
                            st.session_state.reset_input_box = True
                            if st.session_state.problem_state['current_step'] >= len(steps):
                                st.session_state.chat_history.append({
                                    "role": "assistant",
                                    "content": f"The final answer is: {st.session_state.problem_state['final_answer']}",
                                    "timestamp": time.strftime("%H:%M"),
                                    "requires_input": False
                                })

                                with st.spinner("Generating problem summary..."):
                                    summary = st.session_state.solver.generate_problem_summary(
                                        st.session_state.problem_state['solution']
                                    )
                                st.session_state.chat_history.append({
                                    "role": "assistant",
                                    "content": summary,
                                    "timestamp": time.strftime("%H:%M"),
                                    "requires_input": False
                                })

                                st.session_state.show_feedback_form = True

                                data = {
                                    "original_problem": st.session_state.problem_state['original_problem'],
                                    "current_step": "complete",
                                    "user_input": user_input,
                                    "correct_answer": expected_answer,
                                    "hint": None,
                                    "final_answer": st.session_state.problem_state['final_answer'],
                                    "problem_summary": summary,
                                    "user_feedback": "Problem completed"
                                }
                                append_data_to_sheet(json.dumps(data))

                            else:
                                next_step_num = st.session_state.problem_state['current_step']
                                next_step = st.session_state.problem_state['steps'][next_step_num]
                                st.session_state.chat_history.append({
                                    "role": "assistant",
                                    "content": f"**Step {next_step_num +1}:** {next_step.instruction}\n\n{next_step.question}",
                                    "timestamp": time.strftime("%H:%M"),
                                    "requires_input": True,
                                    "step_num": next_step_num
                                })
                                st.session_state.problem_state['awaiting_answer'] = True

                            data = {
                                "original_problem": st.session_state.problem_state['original_problem'],
                                "current_step": current_step_index + 1,
                                "user_input": user_input,
                                "correct_answer": expected_answer,
                                "hint": current_step.explanation if current_step.hint_count > 0 else None,
                                "final_answer": st.session_state.problem_state['final_answer'],
                                "problem_summary": None,
                                "user_feedback": f"Incorrect answer (Attempt {current_step.attempt_count})"
                            }
                            append_data_to_sheet(json.dumps(data))

                            if current_step.attempt_count >= 3:
                                data = {
                                    "original_problem": st.session_state.problem_state['original_problem'],
                                    "current_step": current_step_index + 1,
                                    "user_input": user_input,
                                    "correct_answer": expected_answer,
                                    "hint": current_step.explanation,
                                    "final_answer": st.session_state.problem_state['final_answer'],
                                    "problem_summary": None,
                                    "user_feedback": "Max attempts reached, showing solution"
                                }
                                append_data_to_sheet(json.dumps(data))

                        else:
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": f"❌ That's not quite right. {explanation}\nYou have {remaining_attempts} attempt{"s" if remaining_attempts > 1 else ""} left.",
                                "timestamp": time.strftime("%H:%M"),
                                "requires_input": True,
                                "step_num": current_step_index
                            })

                            data = {
                                "original_problem": st.session_state.problem_state['original_problem'],
                                "current_step": current_step_index + 1,
                                "user_input": user_input,
                                "correct_answer": expected_answer,
                                "hint": current_step.explanation if current_step.hint_count > 0 else None,
                                "final_answer": st.session_state.problem_state['final_answer'],
                                "problem_summary": None,
                                "user_feedback": f"Incorrect answer (Attempt {current_step.attempt_count})"
                            }
                            append_data_to_sheet(json.dumps(data))

                except Exception as e:
                    # Add error logging
                    logging.error(f"Error in handle_user_input: {str(e)}")
                    st.error(f"An error occurred: {str(e)}")

        st.rerun()

def main_input_box():
    """
    Shows the text input box where user can add or remove characters.
    The on_change callback updates st.session_state.input_buffer accordingly.
    """
    # If "input_box" or "input_buffer" isn't in session state, initialize it.
    if "input_box" not in st.session_state:
        st.session_state["input_box"] = ""
    if "input_buffer" not in st.session_state:
        st.session_state["input_buffer"] = ""

    def sync_input_buffer():
        # This makes sure input_buffer matches whatever is in input_box
        st.session_state.input_buffer = st.session_state.input_box

    st.text_input(
        label="Your Expression",
        key="input_box",  # ties the widget value to st.session_state.input_box
        on_change=sync_input_buffer
    )

def display_chat_history(chat_container):
    with chat_container:
        for idx, message in enumerate(st.session_state.chat_history):
            with st.chat_message(message["role"]):
                st.markdown(f"<div class='step-indicator'>{message['timestamp']}</div>", unsafe_allow_html=True)
                if message["role"] == "user":
                    st.write(message["content"])
                else:
                    content = message["content"]

                    if "The final answer is:" in content:
                        prefix = content.split("The final answer is:")[0]
                        final_answer = content.split("The final answer is:")[-1].strip()

                        if prefix.strip():
                            st.write(prefix.strip())

                        st.write("The final answer is:")
                        final_answer = final_answer.strip().strip('$')
                        st.latex(final_answer)
                    else:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            parts = content.split("$")
                            for i, part in enumerate(parts):
                                if i % 2 == 0:
                                    if part.strip():
                                        if part.startswith("Step"):
                                            st.markdown(f"**{part}**")
                                        else:
                                            st.write(part.strip())
                                else:
                                    if part.strip():
                                        st.latex(part.strip())

                        with col2:
                            if message.get("requires_input"):
                                step_num = message.get("step_num")
                                if st.session_state.problem_state['steps'] and step_num < len(st.session_state.problem_state['steps']):
                                    step = st.session_state.problem_state['steps'][step_num]
                                    if step.graph_image:
                                        st.image(step.graph_image, caption="Graph for this step")

                if message.get("requires_input"):
                    with st.container():
                        col1 = st.columns(1)[0]  # Create a single column

                        step_num = message.get("step_num")
                        current_step = None
                        if step_num is not None:
                            if st.session_state.problem_state['steps'] and step_num < len(st.session_state.problem_state['steps']):
                                current_step = st.session_state.problem_state['steps'][step_num]
                        else:
                            st.write("Error: Could not determine current step.")
                            continue

                        step_key = f"show_question_input_{step_num}"
                        if step_key not in st.session_state:
                            st.session_state[step_key] = False

                        with col1:
                            # Show Hint
                            if st.button("Show Hint", key=f"hint_{idx}"):
                                # Add check for completed problem
                                if st.session_state.problem_state['steps'] is None or step_num >= len(st.session_state.problem_state['steps']):
                                    st.warning("Cannot show hints for a completed problem.")
                                else:    
                                    remaining_hints = max(0, current_step.hint_limit - current_step.hint_count)
                                    if remaining_hints > 0:
                                        previous_attempts = [msg["content"] for msg in st.session_state.chat_history if msg["role"] == "user" and msg.get("step_num") == step_num]
                                        previous_hints = [msg["content"] for msg in st.session_state.chat_history if msg["role"] == "assistant" and msg.get("step_num") == step_num]
                                        hint = st.session_state.solver.generate_hint(current_step, previous_attempts, previous_hints)
                                        current_step.hint_count += 1

                                        st.session_state.chat_history.append({
                                            "role": "assistant",
                                            "content": hint,
                                            "timestamp": time.strftime("%H:%M"),
                                            "requires_input": False,
                                            "step_num": st.session_state.problem_state['current_step']
                                        })

                                    else:
                                        st.warning("You've reached the maximum number of hints for this step.")

                            for i in range(step_num):
                                prev_key = f"show_question_input_{i}"
                                st.session_state[prev_key] = False

                            # Ask Custom Question button
                            if st.button("Ask Custom Question", key=f"custom_{idx}"):
                                st.session_state[step_key] = not st.session_state[step_key]

                            # Only show input area for the current step
                            if step_num == st.session_state.problem_state['current_step'] and st.session_state[step_key]:
                                with st.form(key=f"ask_custom_form_{idx}"):
                                    user_question = st.text_input(
                                        label="",  # Label hidden
                                        label_visibility="hidden",
                                        key=f"hint_input_{idx}",
                                        placeholder="Need clarification? Ask Raze..."
                                    )
                                    ask_submitted = st.form_submit_button("Ask")

                                if ask_submitted:
                                    previous_attempts = [
                                        msg["content"]
                                        for msg in st.session_state.chat_history
                                        if msg["role"] == "user" and msg.get("step_num") == step_num
                                    ]
                                    previous_hints = [msg["content"]
                                        for msg in st.session_state.chat_history
                                        if msg["role"] == "assistant" and msg.get("step_num") == step_num
                                    ]

                                    answer = st.session_state.solver.answer_custom_question(
                                        current_step,
                                        user_question,
                                        previous_attempts,
                                        previous_hints
                                    )

                                    answer_parts = answer.split("$")
                                    for i, part in enumerate(answer_parts):
                                        if i % 2 == 0:
                                            if part.strip():
                                                st.write(part.strip())
                                        else:
                                            if part.strip():
                                                st.latex(part.strip())

                        if current_step:
                            remaining_hints = max(0, current_step.hint_limit - current_step.hint_count)
                            st.caption(f"Remaining hints: {remaining_hints}")

    if st.session_state.problem_state['steps'] is not None and st.session_state.problem_state['awaiting_answer']:
        if not any(msg.get('requires_input') for msg in st.session_state.chat_history[-1:]):
            current_step_index = st.session_state.problem_state['current_step']
            if current_step_index < len(st.session_state.problem_state['steps']) and st.session_state.problem_state['new_step']:
                step = st.session_state.problem_state['steps'][current_step_index]
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"**Step {current_step_index +1}:** {step.instruction}\n\n{step.question}",
                    "timestamp": time.strftime("%H:%M"),
                    "requires_input": True,
                    "step_num": current_step_index
                })
                st.session_state.problem_state['new_step'] = False
                st.rerun()

            elif current_step_index >= len(st.session_state.problem_state['steps']):
                st.session_state.problem_state = {
                    'original_problem': None,
                    'steps': None,
                    'current_step': 0,
                    'expected_answer': None,
                    'variables': set(),
                    'awaiting_answer': False,
                    'final_answer': None,
                    'solution': None,
                    'new_step': True
                } 