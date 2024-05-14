def make_fixed_length(df, desired_length):
    """
    Modifies the DataFrame to have a fixed length by adding or removing rows.

    Args:
        df: The DataFrame to modify.
        desired_length: The desired length of the DataFrame.

    Returns:
        The modified DataFrame with the specified length.
  """

    while df.shape[0] < desired_length:
        # Generate a random index within valid bounds (excluding start and end)
        random_index = np.random.randint(1, df.shape[0])
        # print(random_index)

        # Calculate the average values of the upper and lower index rows
        upper_row = df.iloc[random_index - 1]
        lower_row = df.iloc[random_index]
        average_row = (upper_row + lower_row) / 2

        # Create a new DataFrame with the average values
        # put in the index of random_index
        new_row_df = pd.DataFrame([average_row.values], columns=df.columns)

        # Insert the new row at the specified random index
        df = pd.concat([df.iloc[:random_index], new_row_df, df.iloc[random_index:]])

        # Reset the index to maintain a continuous integer sequence
        df = df.reset_index(drop=True)


    while df.shape[0] > desired_length:
        # Generate a random index within valid bounds (excluding the last index)
        random_index = np.random.randint(0, df.shape[0] - 1)
        # print(random_index)

        # Remove the row at the specified random index
        df = df.drop(random_index)

        # Reset the index to maintain a continuous integer sequence
        df = df.reset_index(drop=True)

    return df