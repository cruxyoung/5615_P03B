from behave import *
from data_examining.data_fetch import fetch_data
from pandas import DataFrame

@given('a framework')
def step_impl(context):
	pass

@when('input the name of txt dataset')
def step_impl(context):
	context.path_ = 'G:\\2017s2\\5615\\project\\source_datasets\\fruit_data.txt'

@then('the framework load the txt data')
def step_impl(context):
	x = fetch_data(context.path_)
	assert type(x) == type(DataFrame())



@when('input the name of csv dataset')
def step_impl(context):
	context.path_ = 'G:\\2017s2\\5615\\project\\source_datasets\\wholesale.csv'

@then('the framework load the csv data')
def step_impl(context):
	x = fetch_data(context.path_)
	assert type(x) == type(DataFrame())



@when('input the name of xlsx dataset')
def step_impl(context):
	context.path_ = 'G:\\2017s2\\5615\\project\\source_datasets\\test.xlsx'

@then('the framework load the xlsx data')
def step_impl(context):
	x = fetch_data(context.path_)
	assert type(x) == type(DataFrame())
