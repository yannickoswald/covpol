import pytest
import sys
sys.path.append("..")

from code.agent_class import CountryAgent


# Can run code here that all the tests need, e.g. reading data etc.




# This 'fixture' means that other test functions can use the object created here.
# Note: Don't try to run this test, it will be called when running the others that need it
@pytest.fixture()
def create_agent():
    # Creates an agent object that can be passed to other test functions
    agent = CountryAgent()
    yield agent

def test_step(agent):
    """Test the step() function of an agent object"""
    # Here put some tests to check that the step method is doing
    # exactly what you want it to.
    pass
