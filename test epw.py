from SRC.SIM.Weather.epwSummary import summarize_epw, epw_resolution_and_missing


# Weather file is the just Path ot name of the CSV with .epw
weather_file = './SRC/SIM/Defaults/Weather/USA_GA_Atlanta-Hartsfield-Jackson.Intl.AP.722190_TMY3.epw'


# summary = summarize_epw(weather_file)
#
# print(summary["coverage"])
# print(summary["resolution"])
# print(summary["missing_data"]["affected_columns"])


out = epw_resolution_and_missing(weather_file)

print("Resolution:", out["resolution"])
print("Inserted missing timestamps:", out["inserted_missing_timestamps"])
print(out["missing_value_report"]["missing_by_column"].head(10))
print("First entirely-missing row:", out["missing_value_report"]["first_row_entirely_missing"])