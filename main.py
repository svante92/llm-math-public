import streamlit as st
import time
import os
import json
from dotenv import load_dotenv
from llm import MathSolver, MathSolution, Step  # Ensure llm.py is correctly imported
from sheets import append_data_to_sheet  # Ensure sheets.py is correctly imported

# Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")  # Get the API key from the environment

# Configure the Streamlit layout
st.set_page_config(layout="wide", page_title="Interactive Math Solver")

# Initialize session state variables
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'current_expression' not in st.session_state:
    st.session_state.current_expression = ""

if 'problem_state' not in st.session_state:
    st.session_state.problem_state = {
        'original_problem': None,
        'steps': None,
        'current_step': 0,
        'expected_answer': None,
        'variables': set(),
        'awaiting_answer': False,
        'final_answer': None,
        'solution': None  # Store the MathSolution object
    }

# Initialize the MathSolver instance
if 'solver' not in st.session_state:
    st.session_state.solver = MathSolver(API_KEY)  # Use the API key from the environment

# Initialize session state variables for input
if 'input_buffer' not in st.session_state:
    st.session_state.input_buffer = ''

if 'input_box' not in st.session_state:
    st.session_state.input_box = ''

if 'user_input_submitted' not in st.session_state:
    st.session_state.user_input_submitted = False

if 'reset_input_box' not in st.session_state:
    st.session_state.reset_input_box = False

if 'show_feedback_form' not in st.session_state:
    st.session_state.show_feedback_form = False

def update_input_buffer():
    st.session_state.input_buffer = st.session_state.input_box

def reset_problem():
    # Reset all session state variables related to the problem
    st.session_state.chat_history = []
    st.session_state.problem_state = {
        'original_problem': None,
        'steps': None,
        'current_step': 0,
        'expected_answer': None,
        'variables': set(),
        'awaiting_answer': False,
        'final_answer': None,
        'solution': None
    }
    st.session_state.input_buffer = ''
    st.session_state.input_box = ''
    st.session_state.user_input_submitted = False
    st.session_state.reset_input_box = False
    st.session_state.show_feedback_form = False
    st.session_state.user_input = ''
    # Reset any other session state variables as needed

    # Display a success message (optional)
    st.success("Problem has been reset.")

    # Rerun the app to reflect changes
    st.experimental_rerun()


def create_calculator_sidebar():
    """Create the calculator sidebar with all buttons"""
    st.sidebar.header("Advanced Calculator")

    # Advanced operations
    st.sidebar.subheader("Advanced Operations")
    cols = st.sidebar.columns(4)
    advanced_ops = [
        ("d/dx", r"\frac{d}{dx}"),
        ("∫", r"\int"),
        ("∫_a^b", r"\int_{a}^{b}"),
        ("lim", r"\lim_{x \to }"),
        ("Σ", r"\sum_{i=1}^{n}")
    ]
    for i, (op, latex) in enumerate(advanced_ops):
        with cols[i % 4]:
            st.latex(latex)
            if st.button(op, key=f"adv_{op}", use_container_width=True):
                if op == "∫":
                    st.session_state.input_buffer += "∫() dx "  # Indefinite integral template
                elif op == "∫_a^b":
                    st.session_state.input_buffer += "∫_a^b () dx "  # Definite integral template
                elif op == "lim":
                    st.session_state.input_buffer += "lim_{x→} () "  # Limit template
                elif op == "Σ":
                    st.session_state.input_buffer += "Σ_{i=1}^n () "  # Summation template
                else:
                    st.session_state.input_buffer += f" {op} "
                # Update the input box content
                st.session_state.input_box = st.session_state.input_buffer

    # Functions
    st.sidebar.subheader("Functions")
    cols = st.sidebar.columns(3)
    functions = ["sin", "cos", "tan", "log", "exp"]
    for i, func in enumerate(functions):
        with cols[i % 3]:
            st.latex(func)
            if st.button(func, key=f"func_{func}", use_container_width=True):
                st.session_state.input_buffer += f"{func}("
                # Update the input box content
                st.session_state.input_box = st.session_state.input_buffer

    # Clear button
    if st.sidebar.button("Clear", key="clear_expr", use_container_width=True):
        st.session_state.input_buffer = ""
        st.session_state.input_box = ""

    # Add a spacer for better layout
    st.sidebar.markdown("---")

    # Reset Problem button
    if st.sidebar.button("Reset Problem", key="reset_button", use_container_width=True):
        reset_problem()


    # Add a spacer for better layout
    st.sidebar.markdown("---")

    # Keyboard shortcuts info
    with st.sidebar.expander("Keyboard Shortcuts"):
        st.markdown("""
        - `Enter`: Submit input
        - `Ctrl + Z`: Undo last step
        - `Ctrl + L`: Clear input
        - `Ctrl + H`: Show/hide hints
        """)

def main():
    # Add LaTeX support CSS
    st.markdown("""
        <style>
        .katex-html {
            display: none;
        }
        .correct-answer {
            background-color: #d4edda;
            color: #155724;
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
        }
        .incorrect-answer {
            background-color: #f8d7da;
            color: #721c24;
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
        }
        .validation-feedback {
            font-size: 0.9em;
            margin-top: 5px;
        }
        .step-instruction {
            font-weight: bold;
            color: #0066cc;
        }
        .calculator-button {
            margin: 2px;
            min-height: 40px;
        }

        /* Center the input container */
        .input-container {
            display: flex;
            justify-content: center;
            align-items: flex-start;
            padding: 10px;
        }

        /* Popup container */
        .popup-overlay {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.5);
            z-index: 1000;
        }

        .popup-content {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            width: 400px;
            max-width: 90%;
            z-index: 1001;
        }

        .popup-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }

        .popup-close {
            cursor: pointer;
            font-size: 24px;
            color: #666;
        }

        .popup-buttons {
            display: flex;
            gap: 10px;
            margin-top: 15px;
        }

        .popup-button {
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-weight: 500;
            flex: 1;
        }

        .quick-hint {
            background-color: #ff4b4b;
            color: white;
        }

        .ask-question {
            background-color: #0066cc;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

    # Title in main area
    st.title("Interactive Math Solver")

    # Create calculator sidebar
    create_calculator_sidebar()

    # Handle resetting the input box before rendering the input widget
    if st.session_state.reset_input_box:
        st.session_state.input_box = ''
        st.session_state.reset_input_box = False

    # Main chat container
    chat_container = st.container()

    # Input area at the bottom without a form
    with st.container():
        # Input box with on_change callback
        user_input = st.text_input(
            "Your Input:",
            value=st.session_state.input_buffer,
            key='input_box',
            on_change=update_input_buffer
        )
        # Submit button
        submit_button = st.button('Submit', key='submit_button')

    # If submit button is clicked, set user_input_submitted to True
    if submit_button:
        st.session_state.user_input_submitted = True

    # Process the input when user_input_submitted is True
    if st.session_state.user_input_submitted:
        # Reset the flag
        st.session_state.user_input_submitted = False
        # Get the user input
        user_input = st.session_state.input_box
        st.session_state.user_input = user_input

        # Process the input
        if user_input:
            # Check if we're awaiting the problem or an answer to a step
            if st.session_state.problem_state['steps'] is None:
                # User has input the original problem
                st.session_state.problem_state['original_problem'] = user_input

                # Process the problem using MathSolver
                with st.spinner("Processing the problem..."):
                    try:
                        # Use the solver from session state
                        solution = st.session_state.solver.get_math_solution(user_input)
                        st.session_state.problem_state['steps'] = solution.steps
                        st.session_state.problem_state['final_answer'] = solution.final_answer
                        st.session_state.problem_state['current_step'] = 0
                        st.session_state.problem_state['awaiting_answer'] = True
                        st.session_state.problem_state['solution'] = solution  # Store the solution

                        # Add assistant's message introducing the problem
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": f"Let's solve this problem step by step: {solution.original_problem}",
                            "timestamp": time.strftime("%H:%M"),
                            "requires_input": False
                        })

                        # Present the first step
                        current_step = st.session_state.problem_state['current_step']
                        step = st.session_state.problem_state['steps'][current_step]
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": f"**Step {current_step +1}:** {step.instruction}\n\n{step.question}",
                            "timestamp": time.strftime("%H:%M"),
                            "requires_input": True,
                            "step_num": current_step
                        })

                        # Clear user input
                        st.session_state.user_input = ''
                        st.session_state.input_buffer = ''
                        st.session_state.reset_input_box = True
                        st.session_state.problem_state['awaiting_answer'] = True

                        # Save initial problem data
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
                        st.error(f"Error processing problem: {str(e)}")
            else:
                # User has input an answer to a step
                current_step_index = st.session_state.problem_state['current_step']
                steps = st.session_state.problem_state['steps']
                current_step = steps[current_step_index]
                expected_answer = current_step.answer

                # Use the validation function
                try:
                    is_correct = st.session_state.solver.validate_step_answer_llm(user_input, expected_answer)
                except Exception as e:
                    is_correct = False

                # Record the attempt
                current_step.attempt_count += 1
                current_step.user_attempts.append({
                    'attempt_number': current_step.attempt_count,
                    'user_answer': user_input,
                    'is_correct': is_correct
                })

                # Add user's answer to chat history
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": user_input,
                    "timestamp": time.strftime("%H:%M"),
                    "step_num": current_step_index
                })

                if is_correct:
                    # Correct answer
                    current_step.user_correct = True  # Update user_correct attribute
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"✅ Correct! {current_step.explanation}",
                        "timestamp": time.strftime("%H:%M"),
                        "requires_input": False,
                        "step_num": current_step_index
                    })
                    # Move to next step
                    st.session_state.problem_state['current_step'] += 1
                    st.session_state.user_input = ''
                    st.session_state.input_buffer = ''
                    st.session_state.reset_input_box = True
                    if st.session_state.problem_state['current_step'] >= len(steps):
                        # All steps completed
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": f"Great job! The final answer is: {st.session_state.problem_state['final_answer']}",
                            "timestamp": time.strftime("%H:%M"),
                            "requires_input": False
                        })

                        # Generate and display problem summary
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

                        # Set flag to show feedback form
                        st.session_state.show_feedback_form = True

                        # Save final summary data
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
                        # Present next step
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

                    # Save correct answer interaction
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
                    # Incorrect answer
                    remaining_attempts = 3 - current_step.attempt_count

                    if current_step.attempt_count >= 3:
                        # Provide the correct answer and move to next step
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": f"❌ That's not quite right. The correct answer is: {expected_answer}",
                            "timestamp": time.strftime("%H:%M"),
                            "requires_input": False,
                            "step_num": current_step_index
                        })
                        current_step.user_correct = False  # Update user_correct attribute

                        # Move to next step
                        st.session_state.problem_state['current_step'] += 1
                        st.session_state.user_input = ''
                        st.session_state.input_buffer = ''
                        st.session_state.reset_input_box = True
                        if st.session_state.problem_state['current_step'] >= len(steps):
                            # All steps completed
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": f"The final answer is: {st.session_state.problem_state['final_answer']}",
                                "timestamp": time.strftime("%H:%M"),
                                "requires_input": False
                            })

                            # Generate and display problem summary
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

                            # Set flag to show feedback form
                            st.session_state.show_feedback_form = True

                            # Save final summary data
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
                            # Present next step
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
                    else:
                        # Less than 3 attempts, allow retry
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": f"❌ That's not quite right. Try again! You have {remaining_attempts} attempts left.",
                            "timestamp": time.strftime("%H:%M"),
                            "requires_input": True,
                            "step_num": current_step_index
                        })
                        st.session_state.user_input = ''
                        st.session_state.input_buffer = ''
                        st.session_state.reset_input_box = True

                    # Save incorrect answer interaction
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
                        # Save failed step data
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

        # Reset states on submit
        st.rerun()

    # Display chat history
    with chat_container:
        for idx, message in enumerate(st.session_state.chat_history):
            with st.chat_message(message["role"]):
                st.markdown(f"<div class='step-indicator'>{message['timestamp']}</div>", unsafe_allow_html=True)

                if message["role"] == "user":
                    # Display user messages as regular text
                    st.write(message["content"])
                else:
                    content = message["content"]

                    # Handle the final answer message differently
                    if "The final answer is:" in content:
                        prefix = content.split("The final answer is:")[0]
                        final_answer = content.split("The final answer is:")[-1].strip()

                        # Display the prefix if it exists
                        if prefix.strip():
                            st.write(prefix.strip())

                        # Display "The final answer is:" and the LaTeX expression
                        st.write("The final answer is:")
                        # Remove any existing $ signs and display as LaTeX
                        final_answer = final_answer.strip().strip('$')
                        st.latex(final_answer)
                    else:
                        # Use columns to align text and graph
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            # Split the content into text and LaTeX parts
                            parts = content.split("$")
                            for i, part in enumerate(parts):
                                if i % 2 == 0:  # Regular text
                                    if part.strip():  # Only display if not empty
                                        if part.startswith("Step"):
                                            st.markdown(f"**{part}**")
                                        else:
                                            st.write(part.strip())
                                else:  # LaTeX expression
                                    if part.strip():  # Only display if not empty
                                        st.latex(part.strip())

                        # Display graph image if available and it's a step question
                        with col2:
                            if message.get("requires_input"):
                                step_num = message.get("step_num")
                                if st.session_state.problem_state['steps'] and step_num < len(st.session_state.problem_state['steps']):
                                    step = st.session_state.problem_state['steps'][step_num]
                                    if step.graph_image:
                                        st.image(step.graph_image, caption="Graph for this step")

                # Right-aligned buttons
                if message.get("requires_input"):
                    with st.container():
                        col1, col2 = st.columns([7, 3])

                        # Get step number
                        step_num = message.get("step_num")
                        current_step = None
                        if step_num is not None:
                            if st.session_state.problem_state['steps'] and step_num < len(st.session_state.problem_state['steps']):
                                current_step = st.session_state.problem_state['steps'][step_num]
                        else:
                            st.write("Error: Could not determine current step.")
                            continue

                        # Initialize session state for this step if needed
                        step_key = f"show_question_input_{step_num}"
                        if step_key not in st.session_state:
                            st.session_state[step_key] = False

                        with col1:
                            if st.button("Show Hint", key=f"hint_{idx}"):
                                remaining_hints = max(0, 3 - current_step.hint_count)
                                if remaining_hints > 0:
                                    hint = current_step.explanation
                                    current_step.hint_count += 1

                                    # Display hint with proper LaTeX formatting
                                    hint_parts = hint.split("$")
                                    for i, part in enumerate(hint_parts):
                                        if i % 2 == 0:  # Regular text
                                            if part.strip():
                                                st.write(part.strip())
                                        else:  # LaTeX expression
                                            if part.strip():
                                                st.latex(part.strip())
                                else:
                                    st.warning("You've reached the maximum number of hints for this step.")

                        with col2:
                            if st.button("Ask Custom Question", key=f"custom_{idx}"):
                                st.session_state[step_key] = not st.session_state[step_key]

                            if st.session_state[step_key]:
                                # Get previous attempts for this step
                                previous_attempts = [
                                    msg["content"]
                                    for msg in st.session_state.chat_history
                                    if msg["role"] == "user" and
                                    msg.get("step_num") == step_num
                                ]

                                user_question = st.text_input(
                                    "",  # Label hidden
                                    key=f"hint_input_{idx}",
                                    placeholder="Need clarification? Ask Raze..."
                                )

                                if st.button("Ask", key=f"ask_{idx}"):
                                    remaining_hints = max(0, 3 - current_step.hint_count)
                                    if remaining_hints > 0:
                                        hint = st.session_state.solver.generate_custom_hint(
                                            current_step,
                                            user_question,
                                            previous_attempts
                                        )
                                        current_step.hint_count += 1

                                        # Display hint with proper LaTeX formatting
                                        hint_parts = hint.split("$")
                                        for i, part in enumerate(hint_parts):
                                            if i % 2 == 0:  # Regular text
                                                if part.strip():
                                                    st.write(part.strip())
                                            else:  # LaTeX expression
                                                if part.strip():
                                                    st.latex(part.strip())
                                    else:
                                        st.warning("You've reached the maximum number of hints for this step.")

                        # Display hint count
                        if current_step:
                            remaining_hints = max(0, 3 - current_step.hint_count)
                            st.caption(f"Remaining hints: {remaining_hints}")

    # If we have steps and awaiting user's answer, but haven't presented the current step yet
    if st.session_state.problem_state['steps'] is not None and st.session_state.problem_state['awaiting_answer']:
        # Check if the last message from assistant requires input
        if not any(msg.get('requires_input') for msg in st.session_state.chat_history[-1:]):
            # Present current step if not already presented
            current_step_index = st.session_state.problem_state['current_step']
            if current_step_index < len(st.session_state.problem_state['steps']):
                step = st.session_state.problem_state['steps'][current_step_index]
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"**Step {current_step_index +1}:** {step.instruction}\n\n{step.question}",
                    "timestamp": time.strftime("%H:%M"),
                    "requires_input": True,
                    "step_num": current_step_index
                })
                st.rerun()
            else:
                # No more steps; reset the problem state
                st.session_state.problem_state = {
                    'original_problem': None,
                    'steps': None,
                    'current_step': 0,
                    'expected_answer': None,
                    'variables': set(),
                    'awaiting_answer': False,
                    'final_answer': None,
                    'solution': None
                }

    # Display feedback form after problem completion
    if st.session_state.show_feedback_form:
        st.markdown("---")
        st.markdown("### We'd love your feedback!")
        with st.form(key='feedback_form'):
            name = st.text_input("Name")
            email = st.text_input("Email")
            feedback = st.text_area("Share your experience and thoughts using RazeMath!")
            submit_feedback = st.form_submit_button("Submit Feedback")

            if submit_feedback and name and email and feedback:
                # Save feedback to sheet
                feedback_data = {
                    "original_problem": st.session_state.problem_state['original_problem'],
                    "current_step": "feedback",
                    "user_input": None,
                    "correct_answer": None,
                    "hint": None,
                    "final_answer": st.session_state.problem_state['final_answer'],
                    "problem_summary": None,
                    "user_feedback": f"Name: {name}, Email: {email}, Feedback: {feedback}"
                }
                append_data_to_sheet(json.dumps(feedback_data))

                # Reset states
                st.session_state.show_feedback_form = False
                st.session_state.problem_state = {
                    'original_problem': None,
                    'steps': None,
                    'current_step': 0,
                    'expected_answer': None,
                    'variables': set(),
                    'awaiting_answer': False,
                    'final_answer': None,
                    'solution': None
                }
                st.success("Thank you for your feedback!")
                st.rerun()

if __name__ == "__main__":
    main()