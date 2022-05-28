from valentine import valentine_match, valentine_metrics

def match_on_function(df1, df2, matching_function, args):
	return valentine_match(df1, df2, matching_function(**args))

TRAVERSAL_MATCH = match_on_function