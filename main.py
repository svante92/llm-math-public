from ui.sidebar import create_calculator_sidebar
from ui.chat import display_chat_history, handle_user_input
from ui.feedback import display_feedback_form
from llm import MathSolver
from utils import load_environment_variables
import streamlit as st

def main():
    # Load environment variables
    API_KEY = load_environment_variables()

    # Initialize the MathSolver instance
    if 'solver' not in st.session_state:
        st.session_state.solver = MathSolver(API_KEY)

    # Initialize session state variables
    if 'reset_input_box' not in st.session_state:
        st.session_state.reset_input_box = False
    if 'show_feedback_form' not in st.session_state:
        st.session_state.show_feedback_form = False
    if 'input_buffer' not in st.session_state:
        st.session_state.input_buffer = ''
    if 'input_box' not in st.session_state:
        st.session_state.input_box = ''
    if 'user_input_submitted' not in st.session_state:
        st.session_state.user_input_submitted = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'problem_state' not in st.session_state:
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

    # Configure the Streamlit layout
    st.set_page_config(layout="wide", page_title="Interactive Math Solver")

    # Create calculator sidebar
    create_calculator_sidebar()

    # Main chat container
    chat_container = st.container()

    # Handle user input and display chat history
    handle_user_input()
    display_chat_history(chat_container)

    # Display feedback form after problem completion
    if st.session_state.show_feedback_form:
        display_feedback_form()

if __name__ == "__main__":
    main()