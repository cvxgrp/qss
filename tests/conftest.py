from pytest import fixture


def pytest_addoption(parser):
    parser.addoption("--_verbose", action="store", default=False)


@fixture()
def _verbose(request):
    return request.config.getoption("--_verbose")
