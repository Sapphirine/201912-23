import pandas_gbq
from django.shortcuts import render
from google.oauth2 import service_account

# Make sure you have installed pandas-gbq at first;
# You can use the other way to query BigQuery.
# please have a look at
# https://cloud.google.com/bigquery/docs/reference/libraries#client-libraries-install-nodejs
# To get your credential

credentials = service_account.Credentials.from_service_account_file(
    '/Users/huangjin/Desktop/hw4_tutorial/jh4137-ca7ccf7c464d.json')


def dashboard(request):
    pandas_gbq.context.credentials = credentials
    pandas_gbq.context.project = "jh4137"

    SQL = "SELECT * FROM `jh4137.project.prediction` "
    df = pandas_gbq.read_gbq(SQL)
    data = {}
    data["data"] = []

    df = df.to_dict('records')  # df:{{},{},{}}
    for item in df:  # item:{Home:..., Away:..., win:..., lose:..., draw:..., prediction:...}
        tmpdata = {"Home": "", "Away": '', "Prediction": "", "Percent": ""}
        tmpdata["Home"] = item.pop("Home")
        tmpdata["Away"] = item.pop("Away")
        tmpdata["Prediction"] = item.pop("Prediction")
        tmpdata["Percent"] = item
        data["data"].append(tmpdata)

    pandas_gbq.context.credentials = credentials
    pandas_gbq.context.project = "jh4137"
    data["history"] = []
    SQL = "SELECT * FROM `jh4137.project.season` "
    df = pandas_gbq.read_gbq(SQL)
    df = df.to_dict('records')
    data["history"].append(df)

    return render(request, 'dashboard.html', data)


