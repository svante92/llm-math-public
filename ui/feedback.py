import streamlit as st
from sheets import append_data_to_sheet
import json

def display_feedback_form():
    st.markdown("---")
    st.markdown("### We'd love your feedback!")
    with st.form(key='feedback_form'):
        name = st.text_input("Name")
        email = st.text_input("Email")
        feedback = st.text_area("Share your experience and thoughts using RazeMath!")
        submit_feedback = st.form_submit_button("Submit Feedback")

        if submit_feedback and name and email and feedback:
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