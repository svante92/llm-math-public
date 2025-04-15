import streamlit as st

def create_calculator_sidebar():
    """Create the calculator sidebar with all buttons"""
    st.sidebar.header("Advanced Calculator")
    st.sidebar.markdown("---")

    if st.sidebar.button("Reset Problem", key="reset_button", use_container_width=True):
        reset_problem()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Calculus")
    # Creating 4 columns for calculus buttons
    cols = st.sidebar.columns(4)
    calculus_ops = [
        ("d/dx", r"\frac{d}{dx}"),
        ("∫", r"\int"),
        ("∫[a, b]", r"\int_{a}^{b}"),
        ("lim", r"\lim_{x \to a}")
    ]
    for i, (op, latex) in enumerate(calculus_ops):
        with cols[i % 4]:
            st.latex(latex)
            if st.button(op, key=f"calc_{op}", use_container_width=True):
                if op == "d/dx":
                    st.session_state.input_buffer += "d/dx()"
                elif op == "∫":
                    st.session_state.input_buffer += "∫() dx"
                elif op == "∫[a, b]":
                    st.session_state.input_buffer += "∫[a, b]() dx"
                elif op == "lim":
                    st.session_state.input_buffer += "lim[x->a]()"
                st.session_state.input_box = st.session_state.input_buffer

    st.sidebar.markdown("---")
    st.sidebar.subheader("Trigonometric")
    # Creating 3 columns for trigonometric buttons
    cols = st.sidebar.columns(3)
    trig_ops = ["sin", "cos", "tan", "arcsin", "arccos", "arctan"]
    for i, func in enumerate(trig_ops):
        with cols[i % 3]:
            st.latex(func)
            if st.button(func, key=f"func_{func}", use_container_width=True):
                st.session_state.input_buffer += f"{func}()"
                st.session_state.input_box = st.session_state.input_buffer

    st.sidebar.markdown("---")
    st.sidebar.subheader("Matrix")
    # Creating matrix operations – using 2 columns for clarity
    matrix_ops = [
        ("Matrix", r"\begin{pmatrix} a & b \\ c & d \end{pmatrix}"),
        ("det", "det()"),
        ("inv", "inv()"),
        ("A^T", "A^T")
    ]
    cols_matrix = st.sidebar.columns(2)
    for i, (op, latex) in enumerate(matrix_ops):
        with cols_matrix[i % 2]:
            st.latex(latex)
            if st.button(op, key=f"matrix_{op}", use_container_width=True):
                if op == "Matrix":
                    # A simple 2×2 matrix placeholder; can be expanded as needed.
                    st.session_state.input_buffer += "[[ , ]; [ , ]]"
                elif op == "det":
                    st.session_state.input_buffer += "det()"
                elif op == "inv":
                    st.session_state.input_buffer += "inv()"
                elif op == "A^T":
                    st.session_state.input_buffer += "^T"
                st.session_state.input_box = st.session_state.input_buffer

    st.sidebar.markdown("---")
    st.sidebar.subheader("Other")
    # Creating Other operations
    other_ops = [
        ("Σ", r"\sum_{i=1}^{n}"),
        ("e^x", "e^x"),
        ("log", "log()")
    ]
    cols_other = st.sidebar.columns(3)
    for i, (op, latex) in enumerate(other_ops):
        with cols_other[i % 3]:
            st.latex(latex)
            if st.button(op, key=f"other_{op}", use_container_width=True):
                if op == "Σ":
                    st.session_state.input_buffer += "Σ[i=1 to n]()"
                elif op == "e^x":
                    st.session_state.input_buffer += "e^()"
                elif op == "log":
                    st.session_state.input_buffer += "log()"
                else:
                    st.session_state.input_buffer += f" {op} "
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
