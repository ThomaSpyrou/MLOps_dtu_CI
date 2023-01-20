import requests


def res_404():
    response = requests.get('https://api.github.com/this-api-should-not-exist')
    print(response.status_code)

    if response.status_code == 200:
        print('Success!')
    elif response.status_code == 404:
        print('Not Found.')


def res_200():
    response = requests.get('https://api.github.com')
    print(response.status_code)

    if response.status_code == 200:
        print('Success!')
    elif response.status_code == 404:
        print('Not Found.')


def call_api():
    response = requests.get(
        'https://api.github.com/search/repositories',
        params={'q': 'requests+language:python'})
    
    print(response)


if __name__ == "__main__":
    res_404()
    res_200()
    call_api()
