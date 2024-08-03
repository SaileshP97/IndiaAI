from utility import extract_data, embedding

if __name__ == "__main__":

    purchase_table_path = "/Users/saileshpanda/Desktop/IndiaAI/PDF/Purchase.pdf"
    redemption_table_path = "/Users/saileshpanda/Desktop/IndiaAI/PDF/Redemption.pdf"

    purchase_table = extract_data(purchase_table_path)
    redemption_table = extract_data(redemption_table_path)

    embedding(purchase_table, redemption_table)