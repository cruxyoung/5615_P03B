Feature: The framework of machine learning

Scenario: input txt data
	Given a framework
	When input the name of txt dataset
	Then the framework load the txt data

Scenario: input csv data
	Given a framework
	When input the name of csv dataset
	Then the framework load the csv data

Scenario: input xlsx data
	Given a framework
	When input the name of xlsx dataset
	Then the framework load the xlsx data

