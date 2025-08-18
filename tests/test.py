import pytest
@pytest.mark.parametrize("dtype", (float, int))
class TestA:

    @pytest.fixture(scope='class', autouse=True)
    def setup_and_teardown(self, dtype):
        print('setup')
        if isinstance(dtype, float):
            self.a = 10
        else:
            self.a = 5
        print(dtype)
        yield
        print('teardown')

    def test1(self, dtype):
        print('test1')