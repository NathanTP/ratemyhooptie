# From https://stackoverflow.com/questions/33084190/default-skip-test-unless-command-line-parameter-present-in-py-test
def pytest_addoption(parser):
    parser.addoption('--external-model', action='store_true', dest="external_model",
                     default=False, help="WARNING: this option costs real human money. Enable tests of external models.")

