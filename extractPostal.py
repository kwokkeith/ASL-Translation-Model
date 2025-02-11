import requests


def pcode_to_data(pcode):
    if int(pcode) % 1000 == 0:
        print(pcode)

    headers = {"Authorization": "Bearer **********************"}
    page = 1
    results = []

    while True:
        try:
            response = requests.get('https://www.onemap.gov.sg/api/common/elastic/search?searchVal={0}&returnGeom=Y&getAddrDetails=Y&pageNum={1}'
                                    .format(pcode, page), headers=headers).json()
        except requests.exceptions.ConnectionError as e:
            print(
                'Fetching {0} failed. Retrying in 2 sec: {1}'.format(pcode, e))

            continue

        results = results + response['results']

        if response['totalNumPages'] > page:
            page = page + 1
        else:
            break

    return results


res = pcode_to_data(768071)
print(res)
