import openai
from typing import List, Dict
from pydantic import BaseModel
import json
import logging
import time

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
    format='%(asctime)s - %(message)s',
    force=True
)

class Step(BaseModel):
    instruction: str
    question: str
    answer: str
    explanation: str
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
3. A concise review of the previous step, including the correct answer. This validates the student's solution, reinforces the critical concept or formula used, and ensures they understand how to apply it moving forward.

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

            start_time = time.time() # Start Time Log

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {
                        "role": "user",
                        "content": prompt.strip(),
                    },
                ],
                stream=False
            )
            end_time = time.time() # End Time Log
            elapsed_time = end_time - start_time
            logging.info(f"TIME LOG: OpenAI Logic request took {elapsed_time:.2f} seconds.")

            print(response.choices[0].message.content)
            return response.choices[0].message.content

        except Exception as e:
            return f"Error solving problem: {str(e)}"

    def get_math_solution(self, problem: str) -> MathSolution:
        """
        Send problem to the assistant and get structured solution steps back
        """
        try:

            start_time = time.time() # Start Time Log for Math Solution

            problem_solution = self.solve_problem(problem)
            print("done0")
            print(problem_solution)
            print("done1")

            logic = time.time() - start_time # OpenAI Steps Total Time
            logging.info(f"TIME LOG: Total time for OpenAI Logic response: {logic:.2f} seconds.")

            start_time_steps = time.time() # Start Time Log for OpenAI Steps Request

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
                                        "explanation": {"type": "string"},
                                        "graph_query": {"type": "string"}
                                    },
                                    "required": ["instruction", "question", "answer", "explanation"],
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
                    {"role": "system", "content": "You are a math teacher who takes a math problem and solution to that problem, and breaks down solution into clear steps."},
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
                
                end_time_steps = time.time() # End Time for Open AI 
                steps_time = end_time_steps - start_time_steps # Total Time for OpenAI
                logging.info(f"TIME LOG: OpenAI Steps request took {steps_time:.2f} seconds.")

                total_time = end_time_steps - start_time # Total Time for entire process
                logging.info(f"TIME LOG: Total processing time: {total_time:.2f} seconds.")
                return math_solution
            else:
                raise Exception("No function call in response")


        except Exception as e:
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
                                "description": "Explain to the student why the answer is correct or incorrect. Do not include any math terms in the answer just plain english Make sure its brief. Also do not reveal the correct answer, only explain why the answer is correct or incorrect. Make sure to address the student directly like youre speaking to them"
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
                        "content": f"Determine if these answers are equivalent based on this question: {step_question}:\nStudent's Answer: {user_answer}\nExpected Answer: {correct_answer}"
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

    def generate_custom_hint(self, step: Step, user_question: str, previous_attempts: List[str] = None) -> str:
        """
        Generate a custom hint based on the user's question about a specific step.
        """
        try:
            if step.hint_count >= 3:
                return "You've reached the maximum number of hints for this step. Try reviewing the previous hints and attempts."

            functions = [
                {
                    "name": "generate_hint",
                    "description": "Generate a helpful hint that addresses the student's question.",
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
            
            prompt = f"""
            Current step instruction: {step.instruction}
            Current step question: {step.question}
            
            Student's question: {user_question}
            
            Previous attempts:
            {previous_attempts_text if previous_attempts else "No previous attempts"}
            
            Provide a helpful hint that:
            1. Addresses the specific step.
            2. States the specific idea or formula needed to solve this step, and includes any relevant numbers from the problem.
            3. If they had previous attempts, explain why they were incorrect
            4. Gives a high-level overview of how to solve the step.
            5. Asks the user a guiding question to help them solve the step.
            6. Does NOT give them the final answer. Never give the answer in the hint, only guide the student in the right direction.
            7. Use proper LaTeX formatting for ALL mathematical expressions ($...$)
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
                step.hint_count += 1
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

