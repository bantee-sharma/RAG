from langchain_text_splitters import CharacterTextSplitter,RecursiveCharacterTextSplitter

rec_split = RecursiveCharacterTextSplitter(chunk_overlap=2,chunk_size=10)

char_slpit = CharacterTextSplitter(separator=" ")

print(char_slpit.split_text('abcdefghijklmnopqrstuvwxyz'))
print(char_slpit.split_text('abcdef ghijklmnop qrstuvwxyz'))
print(rec_split.split_text('abcdef ghijklmnop qrstuvwxyz'))