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
API_KEY = os.getenv("OPENAI_API_KEY")  # Get the API key from the environment
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

    def format_prompt(self, problem: str) -> str:
        return f"""You are an expert math teacher that helps college students understand how to solve problems that appear on their homework and exams. 
You specialize in math questions, but also have extreme expertise in other STEM fields like Computer Science, Statistics, Economics, Biology, Physics, and Chemistry. 
Your main objective is to help students understand the concepts fully so that they can get more out of their classes and do better on exams. 
To do this, break down the problem into steps, where each step explains a critical technique or idea that is necessary for reaching the final answer. 
Each step should be concise, aim to enhance understanding, and be written in a patient, encouraging tone. 
Use the appropriate number of steps, typically between 2-6, and ensure each step logically flows from the last, and has a clear way to arrive at the answer. 
Remember, your goal is to help guide students so that they can learn the concepts effectively and do better in their classes.



For each step, provide:
1. Clear instruction of what needs to be done
2. A question for the student about the specific formula, calculation, or idea needed to complete this step.
3. A concise overview of how to complete the previous step, and the correct answer at the end. 
This validates the student’s methods for solving the previous step, and restates the critical idea, calculation, or formula that was used.

IMPORTANT FORMATTING RULES:
- ALL mathematical expressions MUST be enclosed in LaTeX delimiters ($...$)
- For the final answer, provide it in proper LaTeX format without $ delimiters
- Single variables must be in LaTeX: $x$, $n$
- Powers must be in LaTeX: $x^2$, $n^3$
- Complex expressions must be in LaTeX: $\\frac{{dx}}{{dy}}$, $\\int x^2 dx$
- NEVER use plain text for mathematical symbols or expressions

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

    def get_math_solution(self, problem: str) -> MathSolution:
        """
        Send problem to the assistant and get structured solution steps back
        """
        try:
            functions = [
                {
                    "name": "get_math_solution",
                    "description": "Provide the solution steps for the math problem.",
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
                                        "graph_query": {"type": "string"}  # Add graph query to the schema
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

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a precise math teacher who breaks down problems into clear steps."},
                    {"role": "user", "content": self.format_prompt(problem)}
                ],
                functions=functions,
                function_call={"name": "get_math_solution"},
                temperature=0.2,
            )

            message = response.choices[0].message

            if message.function_call is not None:
                arguments = message.function_call.arguments
                solution = json.loads(arguments)
                
                # Clean and format the final answer for LaTeX
                final_answer = solution["final_answer"]
                final_answer = final_answer.strip('$')
                if '\\' not in final_answer and '^' in final_answer:
                    final_answer = final_answer.replace('^', '^{') + '}'
                solution["final_answer"] = final_answer
                
                # Process each step to generate graphs if needed
                for step in solution["steps"]:
                    graph_query = step.get("graph_query")
                    print(f"Graph Query for Step: {graph_query}")  # Debug: Print the graph query

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
    
    def validate_step_answer_llm(self, user_answer: str, correct_answer: str) -> bool:
        """
        Use the LLM to compare the user's answer and the expected answer.
        
        Args:
            user_answer (str): The answer provided by the user.
            correct_answer (str): The expected correct answer.
        
        Returns:
            bool: True if the answers are equivalent, False otherwise.
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
                                "description": "Brief explanation of why the answer is correct or incorrect"
                            }
                        },
                        "required": ["is_correct", "explanation"],
                        "additionalProperties": False
                    }
                }
            ]

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a helpful assistant that checks if the student's answer is correct."
                    },
                    {
                        "role": "user", 
                        "content": f"Determine if these answers are equivalent:\nStudent's Answer: {user_answer}\nExpected Answer: {correct_answer}"
                    }
                ],
                functions=functions,
                function_call={"name": "validate_answer"},
                temperature=0.0
            )

            if response.choices[0].message.function_call is not None:
                result = json.loads(response.choices[0].message.function_call.arguments)
                return result["is_correct"]
            else:
                raise Exception("No function call in response")

        except Exception as e:
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
            1. Addresses their specific question
            2. Guides them toward understanding the concept
            3. If they had previous attempts, explain why they were incorrect
            4. Use proper LaTeX formatting for ALL mathematical expressions ($...$)
            5. Focus on the process and understanding, not the final answer
            """

            response = self.client.chat.completions.create(
                model="gpt-4",
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
- Commends the student on the steps they did correctly.
- Gently points out the steps they struggled with.
- Offers advice on areas to review or study further.
- Uses encouraging and supportive language.
- Does not reveal any additional answers or solutions.
"""

            response = self.client.chat.completions.create(
                model="gpt-4o",
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

        # Get solution steps
        solution = solver.get_math_solution(problem)

        # Print the structured solution
        print("\nOriginal Problem:", solution.original_problem)
        print("\nSolution Steps:")
        for i, step in enumerate(solution.steps, 1):
            print(f"\nStep {i}:")
            print("Instruction:", step.instruction)
            print("Question:", step.question)
            print("Expected Answer:", step.answer)
            print("Explanation:", step.explanation)

        print("\nFinal Answer:", solution.final_answer)

    except Exception as e:
        print(f"Error: {str(e)}")

