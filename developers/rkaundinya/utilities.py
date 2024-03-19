def to_gpu(x):
  if torch.cuda.is_available():
    return x.to('cuda')
  
  return x.to('cpu')

def tokenize(text, space = True):
    tokens = []
    for token in re.split("([0-9a-zA-Z'-]+)", text):
        if not space:
            token = re.sub("[ ]+", "", token)
        if not token:
            continue
        if re.search("[0-9a-zA-Z'-]", token):                    
            tokens.append(token)
        else: 
            tokens.extend(token)
    return np.array(tokens)