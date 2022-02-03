# GENETIC ALGORITHM.

import pandas as pd

data = pd.read_csv("data_project.csv")
import pandas as pd
import numpy as np
import csv

# c= input("Please input file name with file extension (.Text)")
user1 = pd.read_csv("data_project.csv")
print("Input to the printer=")
print(user1)
d = data['c']
f = d[0]
a = pd.read_csv("data_project.csv", skiprows=0,
                usecols=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 25, 26, 27, 28, 29,
                         30, 31, 32, 33, 34, 35, 36])
print("colour Selection=")
print(a)
b = np.array(a)
data = pd.read_csv("data_project.csv")
z = data['order_num']  # as a Series
y = len(z)
x = np.zeros((y))
w = np.ones((y))
print("Output=", y, "Posters to be printed")


class colourSchedulingProblem:
    """This class encapsulates the colour Scheduling problem
    """

    def __init__(self, hardConstraintPenalty):
        """
        :param hardConstraintPenalty: the penalty factor for a hard-constraint violation
        """
        self.hardConstraintPenalty = hardConstraintPenalty

        # list of colour:
        self.colour = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

        # colour' respective job preferences - morning, evening, night:
        self.jobPreference = [[1, 0, 0], [1, 1, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1], [1, 1, 1], [0, 1, 1], [1, 1, 1]]

        # min and max number of colour allowed for each job - morning, evening, night:
        self.jobMin = [2, 2, 1, 1]
        self.jobMax = [3, 4, 2, 1]
        #             self.jobMin = (x)
        #             self.jobMax = (w)

        # max jobs per week allowed for each colour
        self.maxJobsPerWeek = 5

        # number of weeks we create a schedule for:
        self.weeks = 1

        # useful values:
        self.jobPerDay = len(self.jobMin)
        self.jobsPerWeek = 8 * self.jobPerDay

    def __len__(self):
        """
        :return: the number of jobs in the schedule
        """
        return len(self.colour) * self.jobsPerWeek * self.weeks

    def getCost(self, schedule):
        """
        Calculates the total cost of the various violations in the given schedule
        ...
        :param schedule: a list of binary values describing the given schedule
        :return: the calculated cost
        """

        if len(schedule) != self.__len__():
            raise ValueError("size of schedule list should be equal to ", self.__len__())

        # convert entire schedule into a dictionary with a separate schedule for each colour:
        colourShiftsDict = self.getColourShifts(schedule)

        # count the various violations:
        consecutiveJobViolations = self.countConsecutiveJobViolations(colourShiftsDict)
        jobsPerWeekViolations = self.countJobsPerWeekViolations(colourShiftsDict)[1]
        colourPerJobViolations = self.countcolourPerJobViolations(colourShiftsDict)[1]
        jobPreferenceViolations = self.countJobPreferenceViolations(colourShiftsDict)

        # calculate the cost of the violations:
        hardConstraintViolations = consecutiveJobViolations + colourPerJobViolations + jobsPerWeekViolations
        softConstraintViolations = jobPreferenceViolations

        return self.hardConstraintPenalty * hardConstraintViolations + softConstraintViolations

    def getColourShifts(self, schedule):
        """
        Converts the entire schedule into a dictionary with a separate schedule for each colour
        :param schedule: a list of binary values describing the given schedule
        :return: a dictionary with each colour as a key and the corresponding jobs as the value
        """
        jobsPerColour = self.__len__() // len(self.colour)
        colourShiftsDict = {}
        jobIndex = 0

        for colour in self.colour:
            colourShiftsDict[colour] = schedule[jobIndex:jobIndex + jobsPerColour]
            jobIndex += jobsPerColour

        return colourShiftsDict

    def countConsecutiveJobViolations(self, colourShiftsDict):
        """
        Counts the consecutive job violations in the schedule
        :param colourShiftsDict: a dictionary with a separate schedule for each colour
        :return: count of violations found
        """
        violations = 0
        # iterate over the jobs of each colour:
        for colourhifts in colourShiftsDict.values():
            # look for two cosecutive '1's:
            for job1, job2 in zip(colourhifts, colourhifts[1:]):
                if job1 == 1 and job2 == 1:
                    violations += 1
        return violations

    def countJobsPerWeekViolations(self, colourShiftsDict):
        """
        Counts the max-jobs-per-week violations in the schedule
        :param colourShiftsDict: a dictionary with a separate schedule for each colour
        :return: count of violations found
        """
        violations = 0
        dailyJobsList = []
        # iterate over the jobs of each colour:
        for colourhifts in colourShiftsDict.values():  # all jobs of a single colour
            # iterate over the jobs of each weeks:
            for i in range(0, self.weeks * self.jobsPerWeek, self.jobsPerWeek):
                # count all the '1's over the week:
                dailyJobs = sum(colourhifts[i:i + self.jobsPerWeek])
                dailyJobsList.append(dailyJobs)
                if dailyJobs > self.maxJobsPerWeek:
                    violations += dailyJobs - self.maxJobsPerWeek

        return dailyJobsList, violations

    def countcolourPerJobViolations(self, colourShiftsDict):
        """
        Counts the number-of-colour-per-job violations in the schedule
        :param colourShiftsDict: a dictionary with a separate schedule for each colour
        :return: count of violations found
        """
        # sum the jobs over all colour:
        totalPerJobList = [sum(job) for job in zip(*colourShiftsDict.values())]

        violations = 0
        # iterate over all jobs and count violations:
        for jobIndex, numOfcolour in enumerate(totalPerJobList):
            dailyJobIndex = jobIndex % self.jobPerDay  # -> 0, 1, or 2 for the 3 jobs per day
            if (numOfcolour > self.jobMax[dailyJobIndex]):
                violations += numOfcolour - self.jobMax[dailyJobIndex]
            elif (numOfcolour < self.jobMin[dailyJobIndex]):
                violations += self.jobMin[dailyJobIndex] - numOfcolour

        return totalPerJobList, violations

    def countJobPreferenceViolations(self, colourShiftsDict):
        """
        Counts the colour-preferences violations in the schedule
        :param colourShiftsDict: a dictionary with a separate schedule for each colour
        :return: count of violations found
        """
        violations = 0
        for colourIndex, jobPreference in enumerate(self.jobPreference):
            # duplicate the job-preference over the days of the period
            preference = jobPreference * (self.jobsPerWeek // self.jobPerDay)
            # iterate over the jobs and compare to preferences:
            jobs = colourShiftsDict[self.colour[colourIndex]]
            for pref, job in zip(preference, jobs):
                if pref == 0 and job == 1:
                    violations += 1

        return violations

    def printScheduleInfo(self, schedule):
        """
        Prints the schedule and violations details
        :param schedule: a list of binary values describing the given schedule
        """
        colourShiftsDict = self.getColourShifts(schedule)

        print("Schedule for each job:")

        for colour in colourShiftsDict:  # all jobs of a single colour
            print(colour, ":", colourShiftsDict[colour])

        print("consecutive job violations = ", self.countConsecutiveJobViolations(colourShiftsDict))
        print()

        dailyJobsList, violations = self.countJobsPerWeekViolations(colourShiftsDict)
        # print("daily Jobs = ", dailyJobsList)
        print("Jobs Per Day Violations = ", violations)
        print()

        totalPerJobList, violations = self.countcolourPerJobViolations(colourShiftsDict)
        # print("colour Per Job = ", totalPerJobList)
        print("colour Per Job Violations = ", violations)
        print()

        jobPreferenceViolations = self.countJobPreferenceViolations(colourShiftsDict)
        print("Job Preference Violations = ", jobPreferenceViolations)
        print()


# testing the class:
def main():
    # create a problem instance:
    colour = colourSchedulingProblem(10)

    randomSolution = np.random.randint(2, size=len(colour))
    print("Random Solution = ")
    print(randomSolution)
    print()

    colour.printScheduleInfo(randomSolution)

    # print("Total Cost = ", colour.getCost(randomSolution))


if __name__ == "__main__":
    main()
