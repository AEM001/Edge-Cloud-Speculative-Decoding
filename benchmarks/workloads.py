"""
Workload prompts for benchmarking.

Group A: Easier natural-language prompts (30 prompts)
- Summarization
- Explain a concept
- Rewrite a short paragraph
- Simple QA

Group B: Harder structured prompts (30 prompts)
- Grade-school math
- Code completion
- Structured reasoning
- JSON-like constrained answer
"""

# Group A: Easier natural-language prompts (30)
EASY_PROMPTS = [
    # Summarization (8)
    "Summarize the following text in one sentence: Machine learning is a subset of artificial intelligence that enables computers to learn from data and improve their performance over time without being explicitly programmed.",
    "Provide a brief summary of climate change.",
    "Summarize the plot of Romeo and Juliet in two sentences.",
    "What is the main idea of democracy?",
    "Summarize the benefits of regular exercise.",
    "Briefly explain what photosynthesis is.",
    "Summarize the history of the internet.",
    "What are the key points of a healthy diet?",
    
    # Explain a concept (8)
    "Explain what gravity is in simple terms.",
    "What is electricity and how does it work?",
    "Explain the concept of supply and demand.",
    "What is a computer virus?",
    "Explain the difference between weather and climate.",
    "What is a blockchain?",
    "Explain how a refrigerator works.",
    "What is artificial intelligence?",
    
    # Rewrite a short paragraph (7)
    "Rewrite this sentence to be more formal: I'm gonna go to the store and get some stuff.",
    "Make this paragraph more concise: The reason why I decided to go to the park was because I wanted to enjoy the nice weather and get some fresh air.",
    "Rewrite this in a friendly tone: Your application has been rejected due to insufficient qualifications.",
    "Make this more professional: Hey, what's up with the project?",
    "Rewrite this to be clearer: The thing that happened was that the person who was supposed to do the task didn't do it.",
    "Make this more engaging: The book was about a guy who goes on a trip.",
    "Rewrite this sentence to emphasize the positive: The meeting wasn't a complete failure.",
    
    # Simple QA (7)
    "What is the capital of France?",
    "Who wrote Romeo and Juliet?",
    "What year did World War II end?",
    "What is the largest planet in our solar system?",
    "Who painted the Mona Lisa?",
    "What is the chemical symbol for water?",
    "How many continents are there on Earth?",
]


# Group B: Harder structured prompts (30)
HARD_PROMPTS = [
    # Grade-school math (8)
    "If a train travels at 60 mph for 2.5 hours, how far does it travel? Show your work.",
    "A rectangle has a perimeter of 30 cm and a length of 10 cm. What is its width?",
    "Solve for x: 3x + 7 = 22",
    "If 5 apples cost $3.75, how much do 8 apples cost?",
    "A pizza is cut into 8 equal slices. If you eat 3 slices, what fraction of the pizza did you eat?",
    "What is 15% of 240?",
    "If a number is multiplied by 4 and then 6 is added, the result is 34. What is the number?",
    "A box contains 24 red marbles and 36 blue marbles. What is the ratio of red to blue marbles?",
    
    # Code completion (8)
    "Complete this Python function to calculate the factorial of a number:\n\ndef factorial(n):\n    if n <= 1:\n        return 1\n    # Your code here",
    "Write a function to check if a string is a palindrome.",
    "Complete this code to sort a list of numbers in descending order.",
    "Write a function that returns the sum of all even numbers in a list.",
    "Complete this JavaScript function to reverse a string.",
    "Write a function to find the maximum element in an array.",
    "Complete this code to implement binary search.",
    "Write a function that merges two sorted lists.",
    
    # Structured reasoning (7)
    "If all cats are mammals, and all mammals are animals, are all cats animals? Explain your reasoning step by step.",
    "A bat and ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Show your reasoning.",
    "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
    "A farmer has 17 sheep. All but 9 die. How many sheep are left? Explain.",
    "If you flip a fair coin 3 times, what is the probability of getting exactly 2 heads?",
    "In a race, you pass the person in second place. What place are you in now?",
    "A man builds a house with all 4 sides facing south. A bear walks by. What color is the bear?",
    
    # JSON-like constrained answer (7)
    "Extract the name and age from this text and format as JSON: 'John is 25 years old and lives in New York.'",
    "Parse this sentence into JSON with fields 'subject', 'verb', 'object': 'The cat chased the mouse.'",
    "Convert this to JSON format: Product: Laptop, Price: $999, Stock: 50",
    "Extract key-value pairs from this text as JSON: 'username=admin, password=secret123, role=administrator'",
    "Format this address as JSON: '1600 Pennsylvania Avenue NW, Washington, DC 20500'",
    "Convert this recipe to JSON: 'Mix 2 cups flour, 1 cup sugar, and 3 eggs. Bake at 350°F for 30 minutes.'",
    "Parse this log entry into JSON: '2024-01-15 10:30:45 ERROR Connection timeout on port 8080'",
]


def get_easy_prompts() -> List[str]:
    """Get all easy prompts (Group A)."""
    return EASY_PROMPTS.copy()


def get_hard_prompts() -> List[str]:
    """Get all hard prompts (Group B)."""
    return HARD_PROMPTS.copy()


def get_all_prompts() -> List[str]:
    """Get all prompts (easy + hard)."""
    return EASY_PROMPTS + HARD_PROMPTS


def get_prompts_by_group(group: str) -> List[str]:
    """
    Get prompts by group.
    
    Args:
        group: 'easy' or 'hard'
    
    Returns:
        List of prompts
    """
    if group == "easy":
        return get_easy_prompts()
    elif group == "hard":
        return get_hard_prompts()
    else:
        raise ValueError(f"Unknown group: {group}. Use 'easy' or 'hard'.")
