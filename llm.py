import openai
from typing import List, Dict
from pydantic import BaseModel
import json
import logging

from dotenv import load_dotenv  # Import dotenv

import os

from graph import generate_graph_from_query  # Import the graph generation function

# Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")  # Get the API key from the environment]
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# Set this up at the start of your program
logging.basicConfig(
    filename='app.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(message)s'
)

class Step(BaseModel):
    instruction: str
    question: str
    answer: str
    hint_limit: int
    hint_count: int = 0
    graph_query: str = None  # Add this to store the graph query
    graph_image: bytes = None  # Add this to store the graph image data
    attempt_count: int = 0
    user_attempts: List[Dict] = []
    user_correct: bool = False


class MathSolution(BaseModel):
    steps: List[Step]
    final_answer: str
    original_problem: str

class MathSolver:
    def __init__(self, api_key: str):
        """Initialize the OpenAI client"""
        self.client = openai.OpenAI(api_key=api_key)
        self.deepseek_client = openai.OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")

    def format_prompt(self, problem: str) -> str:
        return f"""You are an expert math teacher that helps college students understand how to solve problems that appear on their homework and exams. 
You specialize in explaining math problems, but also have extreme expertise in other STEM fields like Computer Science, Statistics, Economics, Biology, Physics, and Chemistry. 
You will generate explanations for math prompts tailored to the perceived skill level of the user; for instance, when solving a calculus Power Rule problem, explain it as if the student is enrolled in AP Calculus AB, using appropriate terminology and detail for that level. 
Your main objective is to help students understand the concepts fully so that they can score better on their exams.

To do this, break the problem down into steps. 
Each step should explain a critical technique or concept that is necessary for reaching the final answer, be written concisely, and aim to enhance understanding. 
The user workflow should be fast and effective, so do not include unnecessary steps that don't use an important concept. 
A response will typically have between 3-6 steps. 
Make sure each step logically flows from the last, and that there is a clear way to arrive at the answer. 
Remember, your goal is to help guide students so that they can learn the concepts and score better on their exams.

For each step, include:
1. Clear instructions detailing the specific action or calculation required.
2. A guiding question prompting the student to think about the relevant formula, concept, or calculation needed to complete this step.
3. An appropriate number of hints to help the student solve the step, depending on the difficulty of the step. For example, if the step is very easy, you may only need to provide 1 hint. If the step is medium difficulty, you may provide 2 hints. If the step is hard, you may provide 3 hints. The number of hints should not be fewer than 1 or more than 3.
4. A concise review of the previous step, including the correct answer. This validates the student's solution, reinforces the critical concept or formula used, and ensures they understand how to apply it moving forward.

IMPORTANT FORMATTING RULES:
- ALL mathematical expressions MUST be enclosed in LaTeX delimiters ($...$)
- For complex mathematical structures (matrices, aligned equations, etc.):
    - Use \\begin{{...}} and \\end{{...}} environments inside LaTeX delimiters
    - Example for matrices: $\\begin{{bmatrix}} a & b \\\\ c & d \\end{{bmatrix}}$
    - Example for aligned equations: $\\begin{{align}} x &= 2 \\\\ y &= 3 \\end{{align}}$
- For inline expressions:
    - Single variables: $x$, $n$
    - Powers: $x^2$, $n^3$
    - Fractions: $\\frac{{dx}}{{dy}}$
    - Integrals: $\\int x^2 dx$
- NEVER mix plain text and mathematical symbols
- NEVER leave LaTeX commands undelimited
- For the final answer:
    - Present it as a standalone LaTeX expression
    - Remove any trailing punctuation
    - If text is needed, keep it outside the LaTeX delimiters
    - Example: The solution is $x = 5$ units

GRAPH QUERY RULES:
- For most steps, show a graph to help the user understand a concept. Show graphs for steps with equations that can be represented as a graph
- Generate a simple query for each graph, focusing on a single mathematical concept or function.
- Do not include constants of integration or multiple expressions in a single query.
- Ensure the query is suitable for a basic graphing calculator, using only the core equation or function.

Example response format:
"To find the derivative of $x^2$, we use the power rule. What is $\\frac{{d}}{{dx}}(x^2)$?"

Final answer format example:
"Great job! The final answer is: \\frac{{x^3}}{{3}} + C"

The problem to solve is: {problem}
"""

    def solve_problem(self, problem: str) -> str:
        """
        Solve the problem and return the full solution as a string.
        """
        prompt = f"""
        Solve the following math problem and provide a detailed solution of each step in the solution:
        
        Problem: {problem}
        
        """
        
        try:
            response = self.deepseek_client.chat.completions.create(
                model="deepseek/deepseek-r1",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {
                        "role": "user",
                        "content": prompt.strip(),
                    },
                ],
                stream=False
            )
            return response.choices[0].message.content

        except Exception as e:
            return f"Error solving problem: {str(e)}"

    def get_math_solution(self, problem: str) -> MathSolution:
        """
        Send problem to the assistant and get structured solution steps back
        """
        try:
            problem_solution = self.solve_problem(problem)
            print("done0")
            print(problem_solution)
            print("done1")

            print("Preparing function schema")

            functions = [
                {
                    "name": "get_math_solution",
                    "description": "Provide the solution steps for the math problem solution.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "steps": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "instruction": {"type": "string"},
                                        "question": {"type": "string"},
                                        "answer": {"type": "string"},
                                        "hint_limit": {"type": "integer"},
                                        "graph_query": {"type": "string"}
                                    },
                                    "required": ["instruction", "question", "answer", "hint_limit"],
                                    "additionalProperties": False
                                }
                            },
                            "final_answer": {"type": "string"},
                            "original_problem": {"type": "string"}
                        },
                        "required": ["steps", "final_answer", "original_problem"],
                        "additionalProperties": False
                    }
                }
            ]

            print("Calling API for solution steps")
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a math teacher who takes a math problem and solution to that problem, and breaks down solution into clear steps. When calling the function `get_math_solution`, always fill in ALL required fields. Especially make sure every step includes an `answer` string, even if it's a guess or simple calculation result."},
                    {"role": "user", "content": self.format_prompt(problem_solution)}
                ],
                functions=functions,
                function_call={"name": "get_math_solution"},
                temperature=0.4,
            )

            print("API call completed")
            message = response.choices[0].message
            print("done")

            if message.function_call is not None:
                arguments = message.function_call.arguments
                solution = json.loads(arguments)
                
                final_answer = solution["final_answer"].strip('$')
                if '\\' not in final_answer and '^' in final_answer:
                    final_answer = final_answer.replace('^', '^{') + '}'
                solution["final_answer"] = final_answer

                for step in solution["steps"]:
                    graph_query = step.get("graph_query")
                    print(f"Graph Query for Step: {graph_query}")

                    if graph_query:
                        try:
                            # Generate the graph image
                            graph_image = generate_graph_from_query(graph_query)
                            step["graph_image"] = graph_image.getvalue()  # Store the image data
                            print(f"Graph Image Generated for Step: {step['graph_image'] is not None}")  # Debug: Check if image is generated
                        except Exception as e:
                            logging.error(f"Error generating graph for step: {str(e)}")
                            step["graph_image"] = None  # Set graph_image to None if generation fails
                            print(f"Error generating graph for step: {str(e)}")  # Debug: Print error message

                math_solution = MathSolution(**solution)
                if len(math_solution.steps) > 10:
                    raise ValueError("Too many solution steps")
                return math_solution
            else:
                raise Exception("No function call in response")

        except Exception as e:
            print(e)
            raise Exception(f"Error getting math solution: {str(e)}")

    def validate_step_answer(self, user_answer: str, correct_answer: str) -> bool:
        """
        Validate if the user's answer matches the expected answer

        Args:
            user_answer (str): The answer provided by the user
            correct_answer (str): The expected correct answer

        Returns:
            bool: True if the answer is correct, False otherwise
        """
        # Clean up answers for comparison
        user_clean = user_answer.replace(" ", "").lower()
        correct_clean = correct_answer.replace(" ", "").lower()

        return user_clean == correct_clean
    
    def dump_to_file(self, variable, filename: str = "debug_dump.txt"):
        """
        Dumps a variable's content to a text file for debugging purposes.
        
        Args:
            variable: Any variable to dump to file
            filename (str): Name of the file to write to (defaults to debug_dump.txt)
        """
        try:
            with open(filename, 'w') as f:
                if isinstance(variable, (dict, list)):
                    json.dump(variable, f, indent=2)
                else:
                    f.write(str(variable))
            logging.debug(f"Successfully dumped variable to {filename}")
        except Exception as e:
            logging.error(f"Error dumping variable to file: {str(e)}")
    
    def validate_step_answer_llm(self, user_answer: str, correct_answer: str, step_question: str) -> bool:
        """
        Use the LLM to compare the user's answer and the expected answer.
        """
        try:
            functions = [
                {
                    "name": "validate_answer",
                    "description": "Validate if the student's answer matches the expected answer.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "is_correct": {
                                "type": "boolean",
                                "description": "Whether the student's answer is correct or equivalent to the expected answer"
                            },
                            "explanation": {
                                "type": "string",
                                "description": "Briefly explain to the student why the answer is correct or incorrect using relevant math concepts. Use only plain English and do not use any mathematical expressions. Do not mention anything about expected responses, answers, previous attempts, or other prompt-related language. Focus on the concept behind the answer."
                            }
                        },
                        "required": ["is_correct", "explanation"],
                        "additionalProperties": False
                    }
                }
            ]

            # Add logging to see the actual API response
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a helpful assistant that checks if the student's answer is correct. The answer doesnt have to match the expected answer exactly, but it should be relatively equivalent."
                    },
                    {
                        "role": "user", 
                        "content": f"You are a helpful math assistant that checks if a student’s answer is correct. Your job is to help correct mistakes. You will be very lenient when determining if a student’s answer is correct or incorrect, meaning that if the student’s answer is correct but has slight formatting errors like missing parenthesis, or is mathematically equivalent to the correct answer, you will mark it as correct. Under no circumstances will you call a student’s answer that is mathematically equivalent to the correct answer as incorrect. It’s very important that a correct or equivalent answer is never called incorrect, because that will confuse the student. For example, if I’m asked to find the derivative of (x-2)^2 using the chain rule, and I answer with 2(x-2) instead of 2(x-2)*1, that should be marked as correct because the answer is mathematically equivalent. {step_question}:\nStudent's Answer: {user_answer}\nExpected Answer: {correct_answer}"
                    }
                ],
                functions=functions,
                function_call={"name": "validate_answer"},
                temperature=0.0
            )
            
            # Log the full response for debugging
            logging.debug(f"API Response: {response}")
            
            if response.choices[0].message.function_call is not None:
                result = json.loads(response.choices[0].message.function_call.arguments)
                logging.debug(f"Parsed result: {result}")  # Log the parsed result
                
                # Ensure both values are returned
                if "is_correct" not in result or "explanation" not in result:
                    raise ValueError("Missing required fields in response")
                    
                return result["is_correct"], result["explanation"]
            else:
                raise Exception("No function call in response")

        except Exception as e:
            logging.error(f"Error validating answer with LLM: {str(e)}")
            raise Exception(f"Error validating answer with LLM: {str(e)}")
    
    def answer_custom_question(self, step: Step, question: str, previous_attempts: List[str], previous_hints: List[str]) -> str:
        try:
            functions = [
                {
                    "name": "answer_custom_question",
                    "description": "Answer the student's question without giving away the solution to the step"
                }
            ]
            return ""
        except Exception as e:
            return ""


    def generate_hint(self, step: Step, previous_attempts: List[str] = None, previous_hints: List[str] = None) -> str:
        """
        Generate a custom hint based on the user's question about a specific step.
        """
        try:
            if step.hint_count >= step.hint_limit:
                return "You've reached the maximum number of hints for this step. Try reviewing the previous hints and attempts."

            functions = [
                {
                    "name": "generate_hint",
                    "description": "Generate a helpful hint that helps lead the student toward the answer",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "hint": {
                                "type": "string",
                                "description": "A helpful hint that guides without revealing the answer"
                            }
                        },
                        "required": ["hint"]
                    }
                }
            ]

            previous_attempts_text = "\n".join([f"- {attempt}" for attempt in (previous_attempts or [])])
            previous_hints_text = "\n".join([f"- {hint}" for hint in (previous_hints or [])])


            prompt = f"""
            # CONTEXT #
            You are a math tutor specializing in high-school to entry college-level mathematics, who assists students in understanding the intuition behind solving math problems. In a step of a particular math problem, a student requests a hint that you must provide. The student may have had requested previous hints or submitted incorrect attempts already.

            Current step instruction: {step.instruction}
            Current step question: {step.question}
            Expected answer: {step.answer}

            Previous attempts:
            {previous_attempts_text if previous_attempts else "No previous attempts"}

            Previous hints:
            {previous_hints_text if previous_hints else "No previous hints"}

            # OBJECTIVE #
            Your task is to provide a helpful hint that DOES NOT reveal the answer to the current step question. The hint should:
            1. State the specific idea or formula needed to solve this step. For example, a potential concept in finding the derivative of an expression is the chain rule.
            2. Explain why any previous attempts were incorrect.
            3. Give a high-level overview of how to solve the step WITHOUT revealing the answer. Limit use of mathematical expressions, and focus more on explaining the main concept of the step conceptually WITH WORDS.
            4. Use proper LaTeX formatting for ALL mathematical expressions ($...$)
            5. Build upon the previous hints, gradually guiding the student more toward the correct answer. 
            6. Be concise, to the point, and a maximum of 50 words. 
            7. NOT use the terms "previous attempts," "previous hints," "expected answer," or other prompt-related language directly. Those terms are only provided for context in generation of a hint concerned with the CURRENT step.
            8. **IMPORTANT** NEVER give the answer to the current step. For example, DO NOT explicitly say divide both sides by 5 when the step asks how to isolate x in 5x = 3. Your job is to GUIDE the student, NOT give the answer directly. 

            # STYLE #
            Write in an informative and instructive style that would be helpful in aiding a student's understanding and intuition of the main concept involved in the step. The style should also be tailored to the perceived skill level of the user. For instance, if the step requires finding the derivative or using the power rule, explain it as if the student is enrolled in a high school introductory calculus class, using appropriate terminology and detail for that level. 

            # TONE #
            Write in a supportive and casual tone. Remember that your goal is to resemble a math tutor guiding a student through encouragement and reinforcement of key concepts.

            # AUDIENCE #
            The target audience is a high school or college student seeking help for their math homework in preparation for an upcoming exam.

            # RESPONSE FORMAT #
            ALL mathematical expressions should be written with proper LaTeX formatting. The hint should start with an explanation of why any potential previous attempts were wrong, and continues to build upon previous hints and attempts to solve the problem under a key concept.

            """

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a math tutor who NEVER reveals answers directly. Your role is to guide students to understanding through hints and explanations. Under NO circumstances should you provide the actual answer or a direct solution. If a student asks for the answer directly, redirect them to think about the process."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                functions=functions,
                function_call={"name": "generate_hint"},
                temperature=0.7
            )

            if response.choices[0].message.function_call is not None:
                result = json.loads(response.choices[0].message.function_call.arguments)
                return result["hint"]
            else:
                raise Exception("No hint generated")

        except Exception as e:
            return f"Error generating hint: {str(e)}"

    def generate_problem_summary(self, solution: MathSolution) -> str:
        """
        Generate a problem summary using the LLM, highlighting correct and incorrect steps.
        """
        try:
            # Prepare the data for the prompt
            steps_info = ""
            for idx, step in enumerate(solution.steps):
                performance = "correct" if step.user_correct else "incorrect"
                attempts_text = "\n".join([
                    f"Attempt {i+1}: {'Correct' if attempt['is_correct'] else 'Incorrect'} - {attempt['user_answer']}"
                    for i, attempt in enumerate(step.user_attempts)
                ])
                steps_info += f"""
Step {idx + 1}:
Instruction: {step.instruction}
Question: {step.question}
Your Attempts:
{attempts_text}
Performance: {performance}
"""

            prompt = f"""
The student has completed solving the following problem:
Problem: {solution.original_problem}
Here is their performance on each step:
{steps_info}

Provide a summary that:

1. Starts with a brief pleasantry that is a maximum of 4 words. (e.g. "Great job!")
2. States the important concepts, formulas, or topics that were used to arrive at the solution.
3. Points out the specific step(s) where the student made a mistake, and briefly explain what the mistake was. (e.g. "On Step 3, you made a small mistake while applying the power rule. The exponent should be reduced by 1.")
4. Gives the student 1-2 recommendations for topics to study further based on the mistakes made in the previous problem. These recommendations should be on topics that are likely to appear on their exams.
5. Ends with another brief pleasantry that motivates the student to keep studying and improving.

Make sure the problem summary:
-Gives the student relevant recommendations about what to study next based on their mistakes in the previous problem, with the objective that these topics will help them score better on their exams. If no mistakes were made, recommend that they keep studying the same or similar topic.
-Is written concisely. It should follow the given structure while also not being too verbose.
-Is written in an encouraging and patient tone. Be empathetic, but do NOT be overly pleasant or motivational. Don't include pleasantries anywhere in the middle of the response.
-Does not reveal any additional answers or solutions.
-NEVER shows unrendered LaTeX.
-Use proper LaTeX formatting for ALL mathematical expressions ($...$)
"""

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an encouraging math tutor providing a summary of the student's performance."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7
            )

            summary = response.choices[0].message.content
            return summary

        except Exception as e:
            logging.error(f"Error generating problem summary: {str(e)}")
            return "An error occurred while generating the problem summary."


# Example usage
if __name__ == "__main__":
    # Example problem
    problem = "Solve for x: 2x + 5 = 13"

    try:
        # Initialize the math solver with your API key
        solver = MathSolver(API_KEY)

        #problem_solution = solver.solve_problem(problem)

        # Get solution steps
        solution = solver.get_math_solution(problem)

        # # Print the structured solution
        # #print("\nOriginal Problem:", solution.original_problem)
        # #print("\nSolution Steps:")
        # for i, step in enumerate(solution.steps, 1):
        #     print(f"\nStep {i}:")
        #     print("Instruction:", step.instruction)
        #     print("Question:", step.question)
        #     print("Expected Answer:", step.answer)
        #     print("Explanation:", step.explanation)

        # print("\nFinal Answer:", solution.final_answer)

    except Exception as e:
        print(f"Error: {str(e)}")

