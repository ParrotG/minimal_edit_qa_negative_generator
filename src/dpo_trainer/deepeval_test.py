from deepeval import evaluate
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase

# Replace this with the actual documents that you are passing as input to your LLM.
context=["Vazquez grew up just outside the city of Monterrey to a doctor and an engineer.Monterrey (] ), is the capital and largest city of the northeastern state of Nuevo León, in Mexico."]

# Replace this with the actual output from your LLM application
actual_output="Monterrey"

test_case = LLMTestCase(
    input="Of the cities Abdiel Vázquez has lived in, which one was the capital of Nuevo Leon?",
    actual_output=actual_output,
    context=context
)
metric = HallucinationMetric(
    threshold=0.5,
    model="gpt-4.1"
    )

# To run metric as a standalone
evaluate(test_cases=[test_case], metrics=[metric])