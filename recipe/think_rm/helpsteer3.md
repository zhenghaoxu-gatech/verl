---
license: cc-by-4.0
language:
- en
- zh
- ko
- fr
- es
- ru
- ja
- de
- it
- pt
- pl
- id
- nl
- vi
pretty_name: HelpSteer3
size_categories:
- 10K<n<100K
tags:
- human-feedback
- reinforcement-learning
configs:
- config_name: preference
  default: true
  data_files:
  - split: train
    path: preference/train.jsonl.gz
  - split: validation
    path: preference/validation.jsonl.gz
- config_name: feedback
  data_files:
  - split: train
    path: feedback/train.jsonl.gz
  - split: validation
    path: feedback/validation.jsonl.gz
- config_name: edit
  data_files:
  - split: train
    path: edit/train.jsonl.gz
  - split: validation
    path: edit/validation.jsonl.gz
- config_name: edit_quality
  data_files:
  - split: train
    path: edit_quality/train.jsonl.gz
  - split: validation
    path: edit_quality/validation.jsonl.gz
---
# HelpSteer3

HelpSteer3 is an open-source dataset (CC-BY-4.0) that supports aligning models to become more helpful in responding to user prompts.

HelpSteer3-Preference can be used to train [Llama 3.3 Nemotron Super 49B v1](https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1) (for Generative RMs) and [Llama 3.3 70B Instruct Models](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) (for Bradley-Terry RMs) to produce Reward Models that score as high as 85.5% on [RM-Bench](https://arxiv.org/abs/2410.16184) and 78.6% on [JudgeBench](https://huggingface.co/spaces/ScalerLab/JudgeBench), which substantially surpass existing Reward Models on these benchmarks.

HelpSteer3-Feedback and Edit can be used to train [Llama 3.3 70B Instruct Models](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) to perform a novel approach to Inference Time Scaling (ITS) for open-ended, general-domain tasks, we achieve as high as 93.4% on [Arena Hard](https://github.com/lmarena/arena-hard-auto), which makes it No. 1 on the benchmark as of 18 Mar 2025.

See details on HelpSteer3-Preference at [https://arxiv.org/abs/2505.11475](https://arxiv.org/abs/2505.11475) and HelpSteer3-Feedback and Edit at [https://arxiv.org/abs/2503.04378](https://arxiv.org/abs/2503.04378).

Models were trained using the open-source [NeMo Aligner](https://github.com/NVIDIA/NeMo-Aligner).

HelpSteer3 is a follow-up to the [HelpSteer](https://huggingface.co/datasets/nvidia/HelpSteer) and [HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) datasets.

## RM-Bench LeaderBoard

As of 15 May 2025, our reward models trained with HelpSteer3-Preference are the top performing reward models on [RM-Bench](https://arxiv.org/abs/2410.16184), an improved variant of RewardBench for measuring Reward Model in Chat, Math, Code and Safety.

| Model  |  Chat | Math | Code | Safety | Easy | Normal | Hard | Overall RM-Bench|
|:-----------------------------|:------|:------|:------|:------|:------|:------|:------|:------|
| **[Llama-3.3-Nemotron-70B-Reward-Multilingual](https://huggingface.co/nvidia/Llama-3.3-Nemotron-70B-Reward-Multilingual)**  | **86.2** | 82.4 | 66.8 | 94.1 | 86.5 | 85.4 | **80.0**	| 82.4 |
| **[Qwen-3-Nemotron-32B-Reward](https://huggingface.co/nvidia/Qwen-3-Nemotron-32B-Reward)**       | 86.1 | 76.1  | 70.2  | 95.2   | 85.1  | 83.4   | 77.3  | 81.9             |
| [Llama-3.1-Nemotron-70B-Reward](https://huggingface.co/nvidia/Llama-3.1-Nemotron-70B-Reward) | 70.7 | 64.3 | 57.4 | 90.3 | **92.2**	| 76.8 | 48.0 | 70.7 |
| [Skywork-Reward-Gemma-2-27B](https://huggingface.co/Skywork/Skywork-Reward-Gemma-2-27B) | 71.8 | 59.2 | 56.6 | 94.3 | 89.6 | 75.4 | 50.0 | 70.5 |
| [Skywork-Reward-Llama-3.1-8B](https://huggingface.co/Skywork/Skywork-Reward-Llama-3.1-8B) | 69.5 | 60.6 |  54.5 |**95.7** | 89.0 | 74.7 | 46.6 | 70.1 | 
| **[Llama-3_3-Nemotron-Super-49B-GenRM-Multilingual](https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-GenRM-Multilingual)**  |  77.2 |  91.9  | 	74.7  | 	92.9  | 	90.7  | 	86.7  | 	75.1  | 	84.2  | 
| __*+ voting@32*__ | 76.3  |  **93.2**  |  **79.0**  | 93.5 |  92.1  |  **88.5**  |  75.9 |  **85.5**  | 

*Note that Skywork-Reward-Llama-3.1-8B was the best performing reward model reported on RM-Bench and we evaluated all other models. * 

## JudgeBench LeaderBoard

As of 15 May 2025, our reward models trained with HelpSteer3-Preference are the top performing reward models on [JudgeBench](https://huggingface.co/spaces/ScalerLab/JudgeBench), a popular benchmark for evaluating LLM-as-a-judge applications relating to General Knowledge, Logical Reasoning, Math and Coding.
 
 | Model  |  Knowl.| Reason.| Math | Code | Overall JudgeBench |
 |:-----------------------------|:------|:------|:------|:------|:------|
 | **[Llama-3.3-Nemotron-70B-Reward](https://huggingface.co/nvidia/Llama-3.3-Nemotron-70B-Reward)**  |70.8 |76.5 | 82.1 | 66.7 |73.7 |
 | **[Qwen-3-Nemotron-32B-Reward](https://huggingface.co/nvidia/Qwen-3-Nemotron-32B-Reward)**      | 70.1   | 67.4    | 78.6 |  83.3 | 72.3               |
 | [Llama-3.1-Nemotron-70B-Reward](https://huggingface.co/nvidia/Llama-3.1-Nemotron-70B-Reward)  |62.3 | 72.5 | 76.8 | 57.1 | 66.9 |
 | [Skywork-Reward-Gemma-2-27B](https://huggingface.co/Skywork/Skywork-Reward-Gemma-2-27B) |  59.7 | 66.3 | 83.9 | 50.0	| 64.3 |
 | [Skywork-Reward-Llama-3.1-8B](https://huggingface.co/Skywork/Skywork-Reward-Llama-3.1-8B) | 59.1 | 64.3 |  76.8 | 50.0 | 62.3 |
 | **[Llama-3_3-Nemotron-Super-49B-GenRM](https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-GenRM)**  | **71.4** | 73.5 | **87.5** | 76.2 | 75.1 |
 | __*+ voting@32*__ |  70.8 | **83.7** |**87.5** | **83.3** | **78.6** |

*Note that Skywork-Reward-Llama-3.1-27B was the best performing reward model reported on JudgeBench and we evaluated all other numbers.* 

## Arena Hard LeaderBoard

As of 18 Mar 2025, augmenting models with the Feedback-Edit Inference Time Scaling (ITS) approach leads to the highest performance on Arena Hard.

The Feedback-Edit Inference Time Scaling system comprise of the following models:

1. [Llama-3.3-Nemotron-70B-Feedback](https://huggingface.co/nvidia/Llama-3.3-Nemotron-70B-Feedback)
2. [Llama-3.3-Nemotron-70B-Edit](https://huggingface.co/nvidia/Llama-3.3-Nemotron-70B-Edit)
3. [Llama-3.3-Nemotron-70B-Select](https://huggingface.co/nvidia/Llama-3.3-Nemotron-70B-Select)


| Model  | Arena Hard (95% CI) |
|:-----------------------------|:----------------|
| [Llama 3.3 Nemotron Super 49B v1](https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1) + **Feedback-Edit ITS** | **93.4 (-1.1, 1.0)** |
| [Llama-3.1-Nemotron-70B-Instruct](https://huggingface.co/nvidia/Llama-3.1-Nemotron-70B-Instruct) + **Feedback-Edit ITS** | 92.7 (-1.2, 0.9) |
| o1-mini-2024-09-12          | 92.0 (-1.2, 1.0)  |
| o1-preview-2024-09-12         | 90.4 (-1.1, 1.3)  |
| [Llama 3.3 Nemotron Super 49B v1](https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1) | 88.3 (-1.6, 1.6) |
| claude-3-5-sonnet-20241022     | 85.2 (-1.4, 1.6) |
| [Llama-3.1-Nemotron-70B-Instruct](https://huggingface.co/nvidia/Llama-3.1-Nemotron-70B-Instruct) | 84.9 (-1.7, 1.8) |


## Preference

HelpSteer3 contains 40,476 preference samples, each containing a domain, language, context, two responses, an overall preference score between the responses as well as individual preferences from up to 3 annotators. Each individual preference contains a preference score in addition to a concise reasoning for their preference in 1-2 sentences. Data is split into 95% train and 5% validation.

**Domain** In HelpSteer3, we collected data spanning four domains: General, STEM, Code and Multilingual. The first three domains were annotated by Scale AI while the last domain was annotated by Translated.

**Language** For Code domain, language refers to the programming language while for the Multilingual domain, it refers to the natural language. In other domains, this field will be English. Please note that the Code domain only includes prompts in English while the Multilingual domain does not contain any Code prompts.

**Context** Many of the samples in HelpSteer3 contain multi-turn prompts, hence the context field which is in a form that is compatible with the OpenAI ChatCompletion Message object.

**Preference Score** Integer from -3 to 3, corresponding to:

-3: Response 1 is much better than Response 2

-2: Response 1 is better than Response 2

-1: Response 1 is slightly better than Response 2

0: Response 1 is about the same as Response 2

1: Response 2 is slightly better than Response 1

2: Response 2 is better than Response 1

3: Response 2 is much better than Response 1


**Single-Turn Example**

```json
{
    "domain": "code",
    "language": "javascript_html_css",
    "context": [
        {
            "role": "user",
            "content": "Write a JavaScript program that reads a string of text from the keyboard and performs some very basic encryption on this string. First, the order of the words in the string should be reversed. Second, the order of the letters within each word should be reversed. Finally, the first letter of each word should be inserted between each pair of letters to generate the output string.\nWhen the encryption processing has been completed for the entire input string, the program should print the whole encrypted string back to the user on the screen.\nYou can assume that the maximum number of words in the input stream will be 20.\nAfter the encrypted string has been printed to screen, the program should stop."
        }
    ],
    "response1": "Here's a JavaScript program that meets your requirements. This program uses the `readline-sync` module to read input from the keyboard. Make sure to install it by running `npm install readline-sync` before executing the code.\n\n```javascript\nconst readlineSync = require('readline-sync');\n\nfunction encryptString(inputString) {\n const words = inputString.split(' ').reverse();\n let encryptedString = '';\n\n for (let i = 0; i < words.length; i++) {\n    const word = words[i];\n    const reversedWord = word.split('').reverse().join('');\n\n    for (let j = 0; j < reversedWord.length - 1; j++) {\n      encryptedString += reversedWord[j] + reversedWord[0];\n    }\n\n    encryptedString += reversedWord[reversedWord.length - 1];\n\n    if (i < words.length - 1) {\n      encryptedString += ' ';\n    }\n  }\n\n  return encryptedString;\n}\n\nconst inputString = readlineSync.question('Please enter a string of text (max 20 words): ');\n\nif (inputString.split(' ').length > 20) {\n console.log('Error: The input string contains more than 20 words.');\n process.exit(1);\n}\n\nconst encryptedString = encryptString(inputString);\nconsole.log(`Encrypted string: ${encryptedString}`);\n```\n\nThis program reads a string from the keyboard, checks if it has more than 20 words, and then encrypts the string according to the specified rules. Finally, it prints the encrypted string to the screen.",
    "response2": "```javascript\nconst readline = require('readline');\n\nconst rl = readline.createInterface({\n  input: process.stdin,\n  output: process.stdout\n});\n\nrl.question('Enter a string of text: ', (inputString) => {\n  const words = inputString.split(' ');\n  const reversedWords = words.reverse();\n  const encryptedWords = reversedWords.map(word => word.split('').reverse().join(''));\n  let encryptedString = '';\n\n  for (let i = 0; i < encryptedWords.length; i++) {\n    encryptedString += encryptedWords[i].charAt(0) + encryptedWords[i].substring(1);\n    if (i < encryptedWords.length - 1) {\n      encryptedString += ' ';\n    }\n  }\n\n  console.log('Encrypted string:', encryptedString);\n  rl.close();\n});\n```\n\n**Explanation:**\n\n1. **Input:**\n   - The code uses the `readline` module to read input from the user.\n   - `rl.question()` prompts the user to enter a string and stores it in the `inputString` variable.\n\n2. **Word Reversal:**\n   - `inputString.split(' ')` splits the input string into an array of words based on spaces.\n   - `words.reverse()` reverses the order of words in the array.\n\n3. **Letter Reversal:**\n   - `reversedWords.map(word => ...)` iterates through each word in the reversed array and applies a function to transform it.\n   - Inside the function:\n     - `word.split('')` splits each word into an array of individual letters.\n     - `reverse()` reverses the order of letters in the array.\n     - `join('')` joins the reversed letters back into a string.\n\n4. **Encryption:**\n   - `encryptedString` is initialized as an empty string to store the encrypted result.\n   - The code iterates through the `encryptedWords` array.\n   - For each word:\n     - `encryptedWords[i].charAt(0)` gets the first letter of the word.\n     - `encryptedWords[i].substring(1)` gets the rest of the letters in the word.\n     - These are concatenated and added to `encryptedString`.\n     - A space is added after each word except the last one.\n\n5. **Output:**\n   - `console.log('Encrypted string:', encryptedString)` prints the final encrypted string to the console.\n   - `rl.close()` closes the readline interface, ending the program.\n\n\n\nLet me know if you have any other questions.",
    "overall_preference": -2,
    "individual_preference": [
        {
            "score": -1,
            "reasoning": "@Response 1 is slightly better than @Response 2. @Response 1 needs some improvements but is closer to the expected result than @Response 2. "
        },
        {
            "score": -3,
            "reasoning": "@Response 1 is much better than @Response 2 because it accurately and thoroughly followed the instruction, providing a complete solution for encrypting a string."
        },
        {
            "score": -2,
            "reasoning": "@Response 1 is better than @Response 2 because the former fully addressed the prompt's requirements with a possibility of minor improvements, whereas @Response 2 missed a part in the requirement of the prompt."
        }
    ]
}
```

**Multi-Turn Example**

```json
{
    "domain": "code",
    "language": "python",
    "context": [
        {
            "role": "user",
            "content": "lets do python coding"
        },
        {
            "role": "assistant",
            "content": " def greet(name):\n      return f\"Hello, {name}!\"\n\nprint(greet(\"Alice\"))\n# Output: Hello, Alice!\n\ndef add(a, b):\n    return a + b\n\nprint(add(3, 4))\n# Output: 7\n\ndef is_even(number):\n    return number % 2 == 0\n\nprint(is_even(5))\n# Output: False\n\ndef factorial(n):\n    if n == 0:\n        return 1\n    else:\n        return n * factorial(n-1)\n\nprint(factorial(5))\n# Output: 120\n```"
        },
        {
            "role": "user",
            "content": "First of all, in this scenario, two programmers are collaborating. Programmer A can do as much work as a per day, and programmer B can do as much work as b. Two programmers always finish n (=a+b) jobs a day. The first input given are three numbers, n a b respectively, separated by spaces. So make the code that gets the three inputs.\n"
        },
        {
            "role": "assistant",
            "content": "7 2\n\nHere's the Python code that solves the problem:\n\n```python\ndef work_rate(a, b):\n    return a / (a + b)\n\ndef jobs_per_day(work_rate):\n    return 1 / work_rate\n\ndef main():\n    n, a, b = map(int, input().split())\n    \n    work_rate_ab = work_rate(a, b)\n    jobs_per_day_ab = jobs_per_day(work_rate_ab)\n    \n    print(jobs_per_day_ab)\n\nmain()\n```\n\nThis code first defines a function `work_rate(a, b)` that calculates the combined work rate of programmers A and B. Then, it defines a function `jobs_per_day(work_rate)` that calculates the number of jobs they can complete in a day based on their combined work rate.\n\nIn the `main()` function, we read the input values `n`, `a`, and `b` from the user, and then calculate and print the number of jobs that can be completed in a day by both programmers working together."
        },
        {
            "role": "user",
            "content": "Continuing the above, the work we've received has a level of complexity. Since programmer A likes to perform complex tasks, and programmer B likes to perform simple tasks, they divide the tasks by setting the appropriate integer x. If the complexity is greater than x, then that work will be given to A. And if the complexity is less than x, then that work will be given to B. The program has to find the possible number of x that allows each person to perform exactly the number of tasks of a and b.\nFor an example, if a=2 and b=3, and the complexity of work is 1,1,1,4,4, then the possible x is 2 or 3. So the output will be 2. \nFor another example, if n=5, a=2 and b=3, and the complexity of work is 6,2,3,100,1, then the possible x is 4 or 5. So the output will be 2."
        },
        {
            "role": "assistant",
            "content": "5\n\nHere's the Python code that solves the problem:\n\n```python\ndef find_possible_x(n, a, b, complexities):\n    count = 0\n    for x in range(1, n+1):\n        a_count = sum(1 for c in complexities if c > x)\n        b_count = sum(1 for c in complexities if c <= x)\n        if a_count == a and b_count == b:\n            count += 1\n    return count\n\nn, a, b = map(int, input().split())\ncomplexities = list(map(int, input().split()))\n\nprint(find_possible_x(n, a, b, complexities))\n```\n\nThis code defines a function `find_possible_x(n, a, b, complexities)` that takes the input values `n`, `a`, `b`, and a list of `complexities`. It then iterates through all possible values of `x` from 1 to `n`, and for each value of `x`, it counts the number of tasks that can be assigned to programmer A and programmer B based on the given complexities.\n\nFinally, the code reads the input values `n`, `a`, `b`, and `complexities` from the user, and then calculates and prints the number of possible values of `x` that satisfy the given conditions."
        },
        {
            "role": "user",
            "content": "At the \"tasks_B = [task for task in tasks if task &lt;= x]\", can you fix &lt;= sign to &lt; ?"
        }
    ],
    "response1": "Yes, you are correct. The correct comparison operator should be `<` instead of `<=`. Here's the corrected code:\n\n```python\ndef find_possible_x(n, a, b, complexities):\n    count = 0\n    for x in range(1, n+1):\n        a_count = sum(1 for c in complexities if c > x)\n        b_count = sum(1 for c in complexities if c < x)\n        if a_count == a and b_count == b:\n            count += 1\n    return count\n\nn, a, b = map(int, input().split())\ncomplexities = list(map(int, input().split()))\n\nprint(find_possible_x(n, a, b, complexities))\n```\n\nThank you for pointing out the mistake.",
    "response2": "Sure! Here's the updated line of code with `<` instead of `<=`:\n\n```python\ntasks_B = [task for task in tasks if task < x]\n```\n\nThis change means that tasks with a complexity equal to `x` will no longer be assigned to programmer B.",
    "overall_preference": -2,
    "individual_preference": [
        {
            "score": -2,
            "reasoning": "@Response 1 is better than @Response 2 because @Response 1 provides a full code example, integrating the required change directly into the main function, making it immediately usable for the user. @Response 2, while technically correct, only addresses part of the problem by showing a single line change without embedding it into the overall code, leaving more work for the user to complete the solution."
        },
        {
            "score": -2,
            "reasoning": "@Response 1 is better than @Response 2. This is because it provides the complete corrected code and addresses the main idea of the prompt, even though it mistakenly apologizes for saying the comparison should be '<' instead of '<='.  In comparison, @Response 2 only shows one line of code and does not modify the original code to fit what the user wants, making it less helpful for understanding the complete code."
        },
        {
            "score": -2,
            "reasoning": "@Response 1 is better than @Response 2 because it was able to address the prompt's request accurately and the code was working as expected to output value as stated in the prompt."
        }
    ]
}
```



## Feedback

HelpSteer3-Feedback contains 40,821 samples, each containing a domain, language, context, and two responses (see Preference section for details). In addition, each response was annotated with up to 3 instances of feedback elaborating upon the overall helpfulness of the response. Annotators are asked to provide feedback of 2-10 sentences (50 - 250 words) that starts with 'The response is \{not / slightly / partially / mostly / perfectly\} helpful'. Data is split into 95% train and 5% validation.

While annotators are generally free to write any feedback regarding the helpfulness of responses, we reminded the annotators of a few specific aspects that are important but possibly missed. For Multilingual data, we specifically encourage the annotators to provide feedback if the response language is not the same as the prompt language or the language the user expects the output in (e.g. for translation/language-learning tasks). For Coding data, we encourage the annotators to provide feedback on whether there are sufficient comments to describe the code and whether code follows best practices recommended by [Google Style Guides](https://google.github.io/styleguide/).


**Example**

```json
{
    "domain": "code",
    "language": "javascript_html_css",
    "context": [
        {
            "role": "user",
            "content": "Write a JavaScript program that reads a string of text from the keyboard and performs some very basic encryption on this string. First, the order of the words in the string should be reversed. Second, the order of the letters within each word should be reversed. Finally, the first letter of each word should be inserted between each pair of letters to generate the output string.\nWhen the encryption processing has been completed for the entire input string, the program should print the whole encrypted string back to the user on the screen.\nYou can assume that the maximum number of words in the input stream will be 20.\nAfter the encrypted string has been printed to screen, the program should stop."
        }
    ],
    "response1": "Here's a JavaScript program that meets your requirements. This program uses the `readline-sync` module to read input from the keyboard. Make sure to install it by running `npm install readline-sync` before executing the code.\n\n```javascript\nconst readlineSync = require('readline-sync');\n\nfunction encryptString(inputString) {\n const words = inputString.split(' ').reverse();\n let encryptedString = '';\n\n for (let i = 0; i < words.length; i++) {\n    const word = words[i];\n    const reversedWord = word.split('').reverse().join('');\n\n    for (let j = 0; j < reversedWord.length - 1; j++) {\n      encryptedString += reversedWord[j] + reversedWord[0];\n    }\n\n    encryptedString += reversedWord[reversedWord.length - 1];\n\n    if (i < words.length - 1) {\n      encryptedString += ' ';\n    }\n  }\n\n  return encryptedString;\n}\n\nconst inputString = readlineSync.question('Please enter a string of text (max 20 words): ');\n\nif (inputString.split(' ').length > 20) {\n console.log('Error: The input string contains more than 20 words.');\n process.exit(1);\n}\n\nconst encryptedString = encryptString(inputString);\nconsole.log(`Encrypted string: ${encryptedString}`);\n```\n\nThis program reads a string from the keyboard, checks if it has more than 20 words, and then encrypts the string according to the specified rules. Finally, it prints the encrypted string to the screen.",
    "response2": "```javascript\nconst readline = require('readline');\n\nconst rl = readline.createInterface({\n  input: process.stdin,\n  output: process.stdout\n});\n\nrl.question('Enter a string of text: ', (inputString) => {\n  const words = inputString.split(' ');\n  const reversedWords = words.reverse();\n  const encryptedWords = reversedWords.map(word => word.split('').reverse().join(''));\n  let encryptedString = '';\n\n  for (let i = 0; i < encryptedWords.length; i++) {\n    encryptedString += encryptedWords[i].charAt(0) + encryptedWords[i].substring(1);\n    if (i < encryptedWords.length - 1) {\n      encryptedString += ' ';\n    }\n  }\n\n  console.log('Encrypted string:', encryptedString);\n  rl.close();\n});\n```\n\n**Explanation:**\n\n1. **Input:**\n   - The code uses the `readline` module to read input from the user.\n   - `rl.question()` prompts the user to enter a string and stores it in the `inputString` variable.\n\n2. **Word Reversal:**\n   - `inputString.split(' ')` splits the input string into an array of words based on spaces.\n   - `words.reverse()` reverses the order of words in the array.\n\n3. **Letter Reversal:**\n   - `reversedWords.map(word => ...)` iterates through each word in the reversed array and applies a function to transform it.\n   - Inside the function:\n     - `word.split('')` splits each word into an array of individual letters.\n     - `reverse()` reverses the order of letters in the array.\n     - `join('')` joins the reversed letters back into a string.\n\n4. **Encryption:**\n   - `encryptedString` is initialized as an empty string to store the encrypted result.\n   - The code iterates through the `encryptedWords` array.\n   - For each word:\n     - `encryptedWords[i].charAt(0)` gets the first letter of the word.\n     - `encryptedWords[i].substring(1)` gets the rest of the letters in the word.\n     - These are concatenated and added to `encryptedString`.\n     - A space is added after each word except the last one.\n\n5. **Output:**\n   - `console.log('Encrypted string:', encryptedString)` prints the final encrypted string to the console.\n   - `rl.close()` closes the readline interface, ending the program.\n\n\n\nLet me know if you have any other questions.",
    "feedback1": [
        "The response is partially helpful. The code makes a mistake when generating the encryption, it should use the first letter of each word to be inserted between each pair of letters to generate the output, and it uses the first letter of the reversed string which is incorrect.  The code runs fine, it should be improved by modifying the generation fo the final output. ",
        "The response is slightly helpful. It provides a partial solution, introducing readline-sync for keyboard input, which aligns with the requirements. However, the implementation is incomplete, missing essential components of the encryption process, such as reversing the word order, reversing individual letters, and inserting the first letter between pairs of letters. Additionally, the response does not enforce the maximum word limit (20), as specified in the prompt, and the incomplete explanation further limits its utility. Clear and complete implementation is necessary to make the response more helpful.",
        "The response is mostly helpful. It used a module called `readline-sync` for ability of input in the console because Javascript doesn't have the ability by default. The code is fully correct. It only allows sentences of a maximum of 20 words as instructed in the prompt. Also, the prompt's requirement is that, after reversing the strings, the first letter should be added between every pair of strings. It leaves some ambiguity because the first letter can be from the reversed word or the original word. The code used reversed word's first letter. So, it could have been better if there were comments for such clarifications."
    ],
    "feedback2": [
        "The response is partially helpful. It provides a good explanation of using the readline module for user input and demonstrates correct techniques for splitting, reversing, and joining strings. However, it fails to fully address the encryption step described in the prompt (inserting the first letter of each word between pairs of letters) and does not implement the 20-word limit validation. These omissions prevent the response from being fully functional and meeting all the requirements. Including these missing elements would significantly improve its overall usefulness.",
        "The response is partially helpful. It provide a JavaScript solution that reads a string from the user, processes it by reversing the order of words and letter within each word, and then prints the encrypted string. Despite the clear and well-explained code, this oversight in the encryption logic means that the solution is incomplete and does noy fully adhere to the given instruction, thus making it only partially helpful.",
        "The response is partially helpful. It used a module called `readline` for the ability to input from the console. It gives the code with no errors but it did not follow all the prompt's requirements. The sentence should be limited to 20 words but there is no check for it. Also, after reversing each word, the first letter of each word should be added between every pair of letters but this was not done. There is a mistake in line 15 of the code. Also, the explanation part of this code also doesn't follow the prompt requirements. It just takes the reversed word and returns it as it is. The code explanation is also lengthy."
    ]
}
```

## Edit

HelpSteer3-Edit contains 14,461 samples, each containing a domain, language, context, original response (see Preference section for details) as well as 3 instances of feedback elaborating upon the overall helpfulness of the original response (see Feedback section for details). In addition, annotators were asked to create an edited response based on this feedback (which is provided by a separate pool of annotators). Annotators also write a short change summary explaining what they have edited in the response. Data is split into 95% train and 5% validation.

**Example**

```json
{
    "domain": "code",
    "language": "sql",
    "context": [
        {
            "role": "user",
            "content": "explain sql joins, views and common table expressions, user defined functions with examples and code in sql"
        },
        {
            "role": "assistant",
            "content": "SQL Joins:\nSQL Joins are used to combine rows from two or more tables based on a related column between them. There are four types of SQL Joins: INNER JOIN, LEFT JOIN, RIGHT JOIN, and FULL OUTER JOIN.\n\n1. INNER JOIN: Returns records that have matching values in both tables.\nExample:\n```vbnet\nSELECT Orders.OrderID, Customers.CustomerName\nFROM Orders\nINNER JOIN Customers\nON Orders.CustomerID = Customers.CustomerID;\n```\n2. LEFT JOIN (or LEFT OUTER JOIN): Returns all records from the left table and the matched records from the right table. If there is no match, the result is NULL on the right side.\nExample:\n```vbnet\nSELECT Customers.CustomerName, Orders.OrderID\nFROM Customers\nLEFT JOIN Orders\nON Customers.CustomerID = Orders.CustomerID;\n```\n3. RIGHT JOIN (or RIGHT OUTER JOIN): Returns all records from the right table and the matched records from the left table. If there is no match, the result is NULL on the left side.\nExample:\n```vbnet\nSELECT Orders.OrderID, Customers.CustomerName\nFROM Orders\nRIGHT JOIN Customers\nON Orders.CustomerID = Customers.CustomerID;\n```\n4. FULL OUTER JOIN: Returns all records when there is a match in either the left or right table. If there is no match, the result is NULL on both sides.\nExample:\n```vbnet\nSELECT Customers.CustomerName, Orders.OrderID\nFROM Customers\nFULL OUTER JOIN Orders\nON Customers.CustomerID = Orders.CustomerID;\n```\nSQL Views:\nSQL Views are virtual tables that do not store data themselves but represent the data stored in other tables. They are created by using a SELECT statement and can be used to simplify complex queries, hide sensitive data, and provide a simpler interface for end-users.\n\nExample:\n```vbnet\nCREATE VIEW [Employee Sales] AS\nSELECT Employees.LastName, Employees.FirstName, Orders.OrderID\nFROM Employees\nINNER JOIN Orders\nON Employees.EmployeeID = Orders.EmployeeID;\n```\nSQL Common Table Expressions (CTEs):\nSQL Common Table Expressions are temporary result sets that are defined within a single execution of a SELECT, INSERT, UPDATE, or DELETE statement. They can be used to simplify complex queries, improve performance, and make code more readable.\n\nExample:\n```vbnet\nWITH Sales_CTE (SalesPersonID, NumberOfOrders)\nAS\n(\nSELECT SalesPersonID, COUNT(OrderID)\nFROM SalesOrderHeader\nGROUP BY SalesPersonID\n)\nSELECT E.EmployeeID, E.FirstName, E.LastName, S.NumberOfOrders\nFROM HumanResources.Employee AS E\nINNER JOIN Sales_CTE AS S\nON E.EmployeeID = S.SalesPersonID\nORDER BY S.NumberOfOrders DESC;\n```\nSQL User-Defined Functions (UDFs):\nSQL User-Defined Functions are custom functions that can be created to perform specific tasks. They can take input parameters, perform calculations or operations, and return a value. UDFs can be used to simplify complex queries, improve performance, and make code more modular.\n\nExample:\n```vbnet\nCREATE FUNCTION dbo.GetFullName (@FirstName NVARCHAR(50), @LastName NVARCHAR(50))\nRETURNS NVARCHAR(101)\nAS\nBEGIN\nDECLARE @FullName NVARCHAR(101)\nSET @FullName = @FirstName + ' ' + @LastName\nRETURN @FullName\nEND;\n```\nUsage:\n```vbnet\nSELECT dbo.GetFullName('John', 'Doe') AS FullName;\n```"
        },
        {
            "role": "user",
            "content": "explain sql user defined functions with examples and code in sql"
        },
        {
            "role": "assistant",
            "content": "SQL User-Defined Functions (UDFs) are custom functions that can be created to perform specific tasks. They can take input parameters, perform calculations or operations, and return a value. UDFs can be used to simplify complex queries, improve performance, and make code more modular.\n\nThere are three types of UDFs in SQL:\n\n1. Scalar UDFs: These functions return a single value and can be used in SELECT, WHERE, and ORDER BY clauses.\nExample:\n```vbnet\nCREATE FUNCTION dbo.GetFullName (@FirstName NVARCHAR(50), @LastName NVARCHAR(50))\nRETURNS NVARCHAR(101)\nAS\nBEGIN\nDECLARE @FullName NVARCHAR(101)\nSET @FullName = @FirstName + ' ' + @LastName\nRETURN @FullName\nEND;\n```\nUsage:\n```vbnet\nSELECT dbo.GetFullName('John', 'Doe') AS FullName;\n```\n2. Inline Table-Valued UDFs: These functions return a table as a result and can be used in the FROM clause of a SELECT statement.\nExample:\n```vbnet\nCREATE FUNCTION dbo.GetEmployeeSales (@EmployeeID INT)\nRETURNS TABLE\nAS\nRETURN\n(\nSELECT Orders.OrderID, Orders.OrderDate, Orders.TotalDue\nFROM Orders\nWHERE Orders.EmployeeID = @EmployeeID\n);\n```\nUsage:\n```vbnet\nSELECT * FROM dbo.GetEmployeeSales(1);\n```\n3. Multi-statement Table-Valued UDFs: These functions also return a table as a result, but they can contain multiple statements and can be used to perform complex operations.\nExample:\n```vbnet\nCREATE FUNCTION dbo.GetEmployeeSales (@EmployeeID INT)\nRETURNS @SalesTable TABLE\n(\nOrderID INT,\nOrderDate DATETIME,\nTotalDue MONEY\n)\nAS\nBEGIN\nINSERT INTO @SalesTable\nSELECT Orders.OrderID, Orders.OrderDate, Orders.TotalDue\nFROM Orders\nWHERE Orders.EmployeeID = @EmployeeID\nRETURN\nEND;\n```\nUsage:\n```vbnet\nSELECT * FROM dbo.GetEmployeeSales(1);\n```\nUDFs can be very useful in simplifying complex queries and improving performance. However, they should be used judiciously, as they can also have a negative impact on performance if not used properly. It is recommended to test the performance of UDFs before using them in production environments."
        },
        {
            "role": "user",
            "content": "explain analytical functions in sql with examples and code in sql"
        },
        {
            "role": "assistant",
            "content": "Analytical functions in SQL are used to perform calculations on a set of rows that are related to the current row. These functions are used to perform complex calculations such as ranking, windowing, and aggregation. Analytical functions are also known as window functions because they operate on a window of rows.\n\nThere are several analytical functions available in SQL, including:\n\n1. ROW\\_NUMBER(): This function assigns a unique row number to each row within a result set.\nExample:\n```vbnet\nSELECT ROW_NUMBER() OVER(ORDER BY EmployeeID) AS RowNumber, EmployeeID, FirstName, LastName\nFROM Employees;\n```\n2. RANK(): This function assigns a rank to each row within a result set based on a specified order.\nExample:\n```vbnet\nSELECT RANK() OVER(ORDER BY TotalSales DESC) AS Rank, EmployeeID, FirstName, LastName, TotalSales\nFROM Employees;\n```\n3. DENSE\\_RANK(): This function is similar to RANK(), but it does not skip any ranks.\nExample:\n```vbnet\nSELECT DENSE_RANK() OVER(ORDER BY TotalSales DESC) AS DenseRank, EmployeeID, FirstName, LastName, TotalSales\nFROM Employees;\n```\n4. NTILE(): This function divides a result set into a specified number of groups and assigns a group number to each row.\nExample:\n```vbnet\nSELECT NTILE(4) OVER(ORDER BY TotalSales DESC) AS Quartile, EmployeeID, FirstName, LastName, TotalSales\nFROM Employees;\n```\n5. LEAD(): This function returns the value of a column from the next row in a result set.\nExample:\n```vbnet\nSELECT EmployeeID, FirstName, LastName, TotalSales, LEAD(TotalSales) OVER(ORDER BY TotalSales DESC) AS NextTotalSales\nFROM Employees;\n```\n6. LAG(): This function returns the value of a column from the previous row in a result set.\nExample:\n```vbnet\nSELECT EmployeeID, FirstName, LastName, TotalSales, LAG(TotalSales) OVER(ORDER BY TotalSales DESC) AS PreviousTotalSales\nFROM Employees;\n```\nAnalytical functions can be very useful in performing complex calculations and aggregations on a set of rows. They can also improve performance by reducing the number of queries required to perform a calculation. However, they should be used judiciously, as they can also have a negative impact on performance if not used properly. It is recommended to test the performance of analytical functions before using them in production environments."
        },
        {
            "role": "user",
            "content": "explain product purchase patterns for airlines with example "
        }
    ],
    "original_response": "Product purchase patterns for airlines can be analyzed using SQL queries to extract and analyze data from a database. Here is an example of how to analyze product purchase patterns for airlines:\n\n```vbnet\n-- Calculate the total number of flights booked by each airline\nSELECT Airline, COUNT(FlightID) AS TotalFlights\nFROM Flights\nGROUP BY Airline;\n\n-- Calculate the total revenue generated by each airline\nSELECT Airline, SUM(Price) AS TotalRevenue\nFROM Flights\nGROUP BY Airline;\n\n-- Calculate the average ticket price for each airline\nSELECT Airline, AVG(Price) AS AverageTicketPrice\nFROM Flights\nGROUP BY Airline;\n\n-- Calculate the number of flights booked by each airline per month\nSELECT Airline, MONTH(FlightDate) AS Month, COUNT(FlightID) AS TotalFlights\nFROM Flights\nGROUP BY Airline, MONTH(FlightDate);\n\n-- Calculate the revenue generated by each airline per month\nSELECT Airline, MONTH(FlightDate) AS Month, SUM(Price) AS TotalRevenue\nFROM Flights\nGROUP BY Airline, MONTH(FlightDate);\n\n-- Calculate the average ticket price for each airline per month\nSELECT Airline, MONTH(FlightDate) AS Month, AVG(Price) AS AverageTicketPrice\nFROM Flights\nGROUP BY Airline, MONTH(FlightDate);\n\n-- Calculate the number of passengers booked by each airline per month\nSELECT Airline, MONTH(FlightDate) AS Month, SUM(Passengers) AS TotalPassengers\nFROM Flights\nGROUP BY Airline, MONTH(FlightDate);\n\n-- Calculate the number of passengers booked by each airline per month and year\nSELECT Airline, YEAR(FlightDate) AS Year, MONTH(FlightDate) AS Month, SUM(Passengers) AS TotalPassengers\nFROM Flights\nGROUP BY Airline, YEAR(FlightDate), MONTH(FlightDate);\n```\nThese SQL queries can be used to extract and analyze data from a database to gain insights into product purchase patterns for airlines. By analyzing the data, airlines can identify trends, optimize their pricing strategies, and improve their overall performance.",
    "edited_response": "To explain product purchase patterns for airlines, it is essential to analyze both customer behavior and the data generated from airline transactions. These patterns can be observed in factors like booking times, preferred destinations, types of services (such as classes of service), and customer segments. By analyzing the data from flight bookings, airlines can gain insights into these behaviors. Below, I will describe some common product purchase patterns and provide SQL queries to analyze them.\n\n1. **Booking Time and Frequency**\nCustomers may have specific booking behaviors based on the time of year, month, or day. Some might book well in advance, while others might book last minute. By analyzing booking dates, airlines can identify peak travel seasons and optimize pricing strategies.\n```sql\n-- Number of flights booked by each airline per month\nSELECT Airline, EXTRACT(MONTH FROM FlightDate) AS Month, COUNT(FlightID) AS TotalFlights\nFROM Flights\nGROUP BY Airline, EXTRACT(MONTH FROM FlightDate);\n\n```\n\n2. **Revenue Analysis**\nRevenue can vary based on the time of booking, customer preferences, and class of service. Analyzing how much revenue each airline generates by month can provide insights into pricing strategies and seasonal changes.\n```sql\n-- Revenue generated by each airline per month \nSELECT Airline, EXTRACT(MONTH FROM FlightDate) AS Month, SUM(Price) AS TotalRevenue\nFROM Flights\nGROUP BY Airline, EXTRACT(MONTH FROM FlightDate);\n\n```\n\n3. **Customer Segments and Preferences**\nUnderstanding customer preferences such as frequent routes, ticket class (economy, business, first class), and preferred flight times helps airlines tailor their offerings. By analyzing which flights are most popular, airlines can optimize flight schedules and tailor promotional offers.\n```sql\n-- Total revenue per class of service (e.g., Economy, Business, First Class)\nSELECT Airline, ClassOfService, SUM(Price) AS TotalRevenue\nFROM Flights\nGROUP BY Airline, ClassOfService;\n```\n\n4. **Seasonal Trends**\nAirlines often see different purchasing patterns in peak travel periods like holidays or summer vacations. Analyzing monthly data helps in understanding how demand changes over time.\n```sql\n-- Number of passengers by month\nSELECT Airline, EXTRACT(MONTH FROM FlightDate) AS Month, SUM(Passengers) AS TotalPassengers\nFROM Flights\nGROUP BY Airline, EXTRACT(MONTH FROM FlightDate);\n```\n\n5. **Customer Booking Lead Time**\nThe time between booking and the flight date is another pattern that can be examined. Some customers may book far in advance, while others book at the last minute. By analyzing the booking lead time, airlines can adjust pricing to maximize revenue.\n```sql\n-- Average lead time (difference between booking and flight date)\nSELECT Airline, AVG(DATEDIFF(FlightDate, BookingDate)) AS AverageLeadTime\nFROM Flights\nGROUP BY Airline;\n```\n\n6. **Customer Behavior on Different Routes**\nMore customers are likely to book flights along specific routes due to a tourist site, business center, or even during a season. The study of the trend of the most sought after routes would enable the airline to use their resources wisely.\n```sql\n-- Number of flights by route\nSELECT Airline, Route, COUNT(FlightID) AS TotalFlights\nFROM Flights\nGROUP BY Airline, Route;\nConclusion\n```\n\nBy analyzing data such as booking patterns, revenue generation, passenger preferences, and seasonal trends, airlines can identify key product purchase behaviors. The provided SQL queries allow airlines to gain actionable insights into these behaviors, enabling them to optimize their operations, tailor marketing strategies, and enhance customer satisfaction.",
    "feedback": [
        "The response is partially helpful. It is short, well-organized, and consistent which make the user's job easier to understand the code in the response and its explanation. However, the response only provided information on what data to extract and how to extract them from the database using SQL queries, but the main prompt requested was to \"explain the product purchase patters for airline with examples\", no examples were provided and moreover the product purchase patterns were not full explained.",
        "The response is partially helpful. It provides useful technical details. specifically SQL queries that could help analyse product purchase patterns for airlines. The response does not meet the core requirement of the prompt, which is to explain the pattern and provide an example. The primary focus of the response is methodology (SQL queries), which is valueable but not sufficient in this context, where an explanation of customer behaviour patterns such as booking time, destination preferences, and class of service was needed.",
        "The response is mostly helpful. It provides a variety of SQL queries that cover multiple aspects of analyzing product purchase patterns, such as total flights, total revenue, average ticket price and number of passengers. The queries are clearly written and well-structured, making them easy to understand and modify if needed. It can further improved by taking care of syntax which might be compatible with other SQL database. MONTH() and YEAR() functions can sometimes lead to performance issue, especially if there are many rows in the Flights table."
    ],
    "change_summary": "The response has been rewritten to improve correctness, coherence and helpfulness. It explains the concept of product purchase patterns by including detailed explanation of key customer behaviours such as booking times, revenue trends, customer preferences and seasonal demand along with relevant SQL queries. These additions directly address the prompt's request. It also replaces the Month() and Year() functions with EXTRACT(), ensuring compatibility across different SQL databases and enhancing performance."
}
```

## Edit-Quality

HelpSteer3-Edit-Quality contains 3,274 samples, each containing a domain, language, context, original response as well as 3 instances of feedback elaborating upon the overall helpfulness of the response (see Preference/Feedback sections for details). However, instead of a single edited_response as in the Edit section, samples in this section each contains a bad_edited_response and a good_edited_response. Data is split into 95% train and 5% validation.

In this context, good/bad refers only to whether the edited_response follows the provided feedback: good edited responses faithfully edit the response based on the provided feedback, while bad edited responses contain changes that were not necessarily based on the provided feedback. It is possible that a bad edited response still meaningfully improves upon the original response, but our purpose here is to train a model that can make changes based only on provided feedback with high fidelity.

We constructed this dataset because we initially encountered a number of edited responses which did not follow the provided feedback, as we did not clearly communicate this requirement to our vendor. However, they were gracious enough to rework some of the samples when we clarified our requirement later on. While we did not initially anticipate the value of 'bad edited responses', our experiments revealed that they can support guiding the edit model to better follow (and only follow) the provided feedback.


**Example**

```json
{
    "domain": "general",
    "language": "english",
    "context": [
        {
            "role": "user",
            "content": "What is Power Gum"
        },
        {
            "role": "assistant",
            "content": "Power Gum is a brand of chewing gum that claims to provide a \"boost\" of energy and focus. \n\nHere's what you need to know:\n\n**Claims:**\n\n* **Energy Boost:** Power Gum contains a blend of natural ingredients like guarana, ginseng, and caffeine, which are known for their potential to increase alertness and energy levels.\n* **Focus Enhancement:** The gum is also marketed as a way to improve concentration and mental clarity.\n* **Natural Ingredients:** Power Gum emphasizes using natural ingredients, although the specific amounts and their effectiveness are not always clearly stated.\n\n**Ingredients:**\n\nThe exact ingredients and their proportions can vary depending on the flavor and region. However, common ingredients include:\n\n* **Gum Base:** Provides the chewing texture.\n* **Sugar:** Provides sweetness.\n* **Artificial Flavorings:** Adds the various flavors.\n* **Caffeine:** Provides a stimulant effect.\n* **Guarana:** A natural source of caffeine and other compounds that may enhance energy and focus.\n* **Ginseng:** A traditional herbal remedy that is believed to have energizing and cognitive-enhancing properties.\n\n**Effectiveness:**\n\nThe effectiveness of Power Gum in providing a \"boost\" of energy and focus is not scientifically proven. While some individuals may experience a temporary increase in alertness and focus, the effects are likely to be short-lived and may vary from person to person.\n\n**Potential Side Effects:**\n\n* **Anxiety:** Caffeine can cause anxiety and jitters in some individuals.\n* **Insomnia:** Consuming caffeine late in the day can interfere with sleep.\n* **Digestive Issues:** Some people may experience digestive discomfort after consuming Power Gum.\n\n**Overall:**\n\nPower Gum is a chewing gum marketed as a way to provide a temporary energy and focus boost. While it may offer some benefits for some individuals, it's important to be aware of the potential side effects and to use it in moderation. \n\n**Important Note:**\n\nIt's always best to consult with a healthcare professional before using any new supplement or making significant changes to your diet or lifestyle."
        },
        {
            "role": "user",
            "content": "Who made it"
        },
        {
            "role": "assistant",
            "content": "Power Gum is made by **Power Gum LLC**, a company based in the United States. \n\nThey are a relatively new company, but they have been gaining popularity for their unique gum flavors and marketing strategies."
        },
        {
            "role": "user",
            "content": "No"
        }
    ],
    "original_response": "You are right! My apologies, I seem to have hallucinated that information. \n\nI don't have access to real-time information, including details about specific companies and their products. My knowledge is based on the data I was trained on, and it seems I made a mistake in this case. \n\nWould you like me to try and find some more reliable information about who makes Power Gum? I could try searching the web for you.",
    "good_edited_response": "Power Gum \u2014 the chewing gum \u2014 is manufactured by **Power Gum LLC**, an American company. However, there is also a type of fishing line called \"Power Gum.\" Also referred to as \"powergum,\" this product is a clear, elastic \"gum,\" or rubber, used by anglers to create stop knots and back stops. Power Gum is particularly useful for absorbing a large weight on a fishing line. It is manufactured by a U.K.-located company, Drennan Tackle International.",
    "bad_edited_response": "My apologies. I don't have any information about the makers of Power Gum. You may want to try an internet search or reaching out directly to the company through their website. \n\n\n\nIf you meant something else by \"Power Gum\", let me know and I'd be happy to help in any way I can.",
    "feedback": [
        "The response is slightly helpful, as The last paragraph \"Would you like me to try and find some more reliable information about who makes Power Gum? I could try searching the web for you.\" may be valid. Other than this the response doesn't align with the prompt at all, mentioning that it hallucinated was very odd as general users would not know what this means. It then mentions that it doesn't have the information available, however, after checking this, the information is not new and has been widely available since the early 2000s.",
        "The response is partially helpful because it identifies it has hallucinated earlier in the conversation, but it does not specify which information is fabricated (Power Gum is actually a fishing accessory, not a candy). The responses make it clear that it does not have access to real-time information, but then asks if the user would like for it to search the internet for answers, which contradicts not only what it just said, but also what the AI is capable of.",
        "The response is slightly helpful because it acknowledges the error and offers a solution by suggesting a web search. However, it could be more concise and proactive, making it more useful. The explanation of the mistake is clear, but the extra phrasing somewhat dilutes its effectiveness."
    ]
}
```


## Using the Huggingface Datasets

```python
from datasets import load_dataset

ds = load_dataset("nvidia/HelpSteer3")  # this is the Preference by default
train = ds["train"]  # len(train) = 38459 (95%)
val = ds["validation"]  # len(val) = 2017 (5%)

feedback_ds = load_dataset("nvidia/HelpSteer3", "feedback")
edit_ds = load_dataset("nvidia/HelpSteer3", "edit")
edit_quality_ds = load_dataset("nvidia/HelpSteer3", "edit_quality")
```


## Data sources

1. Prompts are collected based on user-contributed [ShareGPT](https://huggingface.co/datasets/RyokoAI/ShareGPT52K) and [WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M) prompts. ShareGPT was used for the Code/Multilingual domains while WildChat-1M was used for General/STEM domains.
2. Responses are generated by ~20 different commercially-permissively-licensed LLMs (note: none from proprietary LLM providers such as OpenAI, see full list in [paper](https://arxiv.org/abs/2503.04378)). We generate 2 responses per prompt, each from a different model so as to provide diverse responses.
3. Human annotations were obtained based on the details from their respective sections above (Preference/Feedback/Edit).


## Annotation methodology overview

1. We engaged a select group of contractors via Scale AI and Translated. These contractors were provided with comprehensive guidelines for each task together with some annotated examples.
2. The annotation process involved approximately 7000 human annotators across more than 80 regions: see distribution in [paper](https://arxiv.org/abs/2503.04378). Specific domains (other than General) have higher bars for eligibility such as software engineering experience, degrees from relevant fields, or evidence of language fluency.
3. Every sample (in Preference and Feedback) was independently annotated by a minimum of three annotators and up to five annotators. We release only the three annotations that agree with one another the most to remove outliers, which tend to misunderstand the task or overlook an important aspect of the task.
4. Post-annotations, our vendors and separately our research team performed extensive quality assurance. We filtered away a substantial proportion of the initial data to produce the current dataset, as we optimized for a high bar in terms of data quality.

## Ethical statement

Annotators were contracted by vendors Scale AI and Translated, which completed ethical review prior to the start of data collection. Consent for use of data for model training was obtained from all annotators through our vendors. Both Scale AI and Translated implemented several measures to prevent annotators from using LLMs as part of the annotation workflow.

Scale AI engages the Anker Methodology, GISC Impact Sourcing Standard, and UN Sustainable Development Goals to provide a fair and competitive pay. The specific pay is calculated based on many factors, including the specific project, the specialized skillset and expertise required, regional costs of living and then transparently listed on Scale AI platform. Scale AI also provides multiple channels for questions and support, including 24/7 support teams, community discussion channels with specially trained moderators, and a “speak up” hotline where contractors can report concerns anonymously. Worker concerns can be submitted to and are reviewed by the support team, and pay disputes are reviewed by support specialists trained in this area.

Translated has active annotators on the platform earn at least their local minimum wage, with systems in place to ensure that annotators were paid in a timely manner for all the work they completed. Translated also has measures to avoid unreasonable levels of competition between annotators, as well as overwork. These measures include restricting sign-ups of new annotators as well as matching annotators with jobs on the platform, thereby reducing unpaid labour time spent on looking for jobs. In addition, Translated has active measures to protect annotators from work-related health and safety risks. Translated engages annotators with contracts that are written in clear language, accessible to annotators, consistent with annotators’ terms of engagement on the platform, and that do not contain a clause requiring annotators to waive their right to legal recourse. Annotators can communicate with a human platform representative from Translated and there are officially documented and effective processes for annotators to appeal decisions such as bad reviews and ratings, or disciplinary actions. Translated also has a policy to mitigate the risk of discrimination against annotators by the platform or clients.

## Citation

If you find this dataset useful, please cite the following works:

```bibtex
@misc{wang2025helpsteer3preferenceopenhumanannotatedpreference,
      title={HelpSteer3-Preference: Open Human-Annotated Preference Data across Diverse Tasks and Languages}, 
      author={Zhilin Wang and Jiaqi Zeng and Olivier Delalleau and Hoo-Chang Shin and Felipe Soares and Alexander Bukharin and Ellie Evans and Yi Dong and Oleksii Kuchaiev},
      year={2025},
      eprint={2505.11475},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.11475}, 
}

@misc{wang2025dedicatedfeedbackeditmodels,
      title={Dedicated Feedback and Edit Models Empower Inference-Time Scaling for Open-Ended General-Domain Tasks},
      author={Zhilin Wang and Jiaqi Zeng and Olivier Delalleau and Daniel Egert and Ellie Evans and Hoo-Chang Shin and Felipe Soares and Yi Dong and Oleksii Kuchaiev},
      year={2025},
      eprint={2503.04378},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.04378},
}
```