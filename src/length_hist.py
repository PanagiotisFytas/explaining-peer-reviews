from DataLoader import DataLoader
import matplotlib.pyplot as plt
import numpy as np

data_loader = DataLoader(device='cpu',
                                  remove_stopwords=False,
                                  final_decision='excluded',
                                  pretrained_weights='scibert_scivocab_uncased',
                                  allow_empty=False
                                  )

# embeddings_input = data_loader.read_embeddigns_from_file()
reviews = data_loader.read_reviews_only_text()

lengths = [len(review[0].split()) for review in reviews]
print(np.mean(lengths), np.std(lengths))


cnt = 0
for length in lengths:
    if length > 512:
        cnt += 1

#plt.hist(lengths, bins=50)
less = []
more = []
for length in lengths:
    if length > 512:
        more.append(length)
    else:
        less.append(length)

plt.hist([less, more], bins=range(1,1409, 32), histtype='barstacked', label=['Reviews with less than 512 tokens', 'Reviews with more than 512 tokens'])
plt.xticks(ticks=range(0,1409,128))
plt.legend()
plt.xlabel('Number of Tokens in Review')
plt.savefig('hist.png')

print('Number of review with more that 512 tokens: ', cnt)
print('Percentage of review with more that 512 tokens: ', cnt/len(lengths))

