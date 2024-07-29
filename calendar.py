import calendar
from colorama import Fore, Style, init

# colorama to use colored output
init(autoreset=True)


def colored_calendar(year, month):
    header_color = Fore.GREEN + Style.BRIGHT
    day_color = Fore.YELLOW
    weekend_color = Fore.RED

    # Fetching the calendar for the specified month and year
    month_calendar = calendar.monthcalendar(year, month)
    month_name = calendar.month_name[month]  # Get the full name of the month
    weekday_names = calendar.day_name  # Get the names of the weekdays
    print(header_color + f"{month_name} {year}")
    print(header_color + " ".join(day_color + day[:2] for day in weekday_names))


    for week in month_calendar:
        week_str = ""
        for day in week:
            if day == 0:
                week_str += "   "  # Adding space for empty days (days not in the month)
            else:
                day_str = f"{day:2d}"  # Formating the day with leading space if needed
                # Checking if the day is a weekend
                if calendar.weekday(year, month, day) >= 5:  # Saturday or Sunday
                    week_str += weekend_color + day_str + " "
                else:
                    week_str += day_color + day_str + " "
        # Printing the formatted week
        print(week_str)


colored_calendar(2024, 5)
