import streamlit as st

def create_calculator_sidebar():
    """Create the calculator sidebar with all buttons"""
    st.sidebar.header("Advanced Calculator")
    st.sidebar.markdown("---")

    if st.sidebar.button("Reset Problem", key="reset_button", use_container_width=True):
        reset_problem()

    st.sidebar.markdown("---")
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
                    st.session_state.input_buffer += "∫() dx "
                elif op == "∫_a^b":
                    st.session_state.input_buffer += "∫_a^b () dx "
                elif op == "lim":
                    st.session_state.input_buffer += "lim_{x→} () "
                elif op == "Σ":
                    st.session_state.input_buffer += "Σ_{i=1}^n () "
                else:
                    st.session_state.input_buffer += f" {op} "
                st.session_state.input_box = st.session_state.input_buffer

    st.sidebar.markdown("---")
    st.sidebar.subheader("Functions")
    cols = st.sidebar.columns(3)
    functions = ["sin", "cos", "tan", "log", "e^x"]
    for i, func in enumerate(functions):
        with cols[i % 3]:
            st.latex(func)
            if st.button(func, key=f"func_{func}", use_container_width=True):
                if func == "e^x":
                    st.session_state.input_buffer += "e^("
                else:
                    st.session_state.input_buffer += f"{func}("
                    st.session_state.input_box = st.session_state.input_buffer

    st.sidebar.markdown("---")
    if st.sidebar.button("Clear", key="clear_expr", use_container_width=True):
        st.session_state.input_buffer = ""
        st.session_state.input_box = ""

    st.sidebar.markdown("---")
    with st.sidebar.expander("Keyboard Shortcuts"):
        st.markdown("""
        - `Enter`: Submit input
        - `Ctrl + Z`: Undo last step
        - `Ctrl + L`: Clear input
        - `Ctrl + H`: Show/hide hints
        """)

def reset_problem():
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
    st.success("Problem has been reset.")
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
