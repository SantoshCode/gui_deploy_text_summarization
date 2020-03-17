from model import *

def summarize(text):


    checkpoint = "/home/sant/projects/gui_deploy/latest_trained_model/best_model.ckpt" 

    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        # Load saved model
        loader = tf.train.import_meta_graph(checkpoint + '.meta')
        loader.restore(sess, checkpoint)
        names = []
        [names.append(n.name) for n in loaded_graph.as_graph_def().node]
    names


    #/content/drive/My Drive/trained model v1/trained model v2/best_model.ckpt
# Create your own review or use one from the dataset
    input_sentence = text
    text = text_to_seq(input_sentence)
    random = np.random.randint(0,len(clean_texts))
#input_sentence = clean_texts[random]
#text = text_to_seq(clean_texts[random])

    checkpoint = "/home/sant/projects/gui_deploy/latest_trained_model/best_model.ckpt"
	

 
    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
        loader = tf.train.import_meta_graph(checkpoint + '.meta')
        loader.restore(sess, checkpoint)

        input_data = loaded_graph.get_tensor_by_name('input:0')
        logits = loaded_graph.get_tensor_by_name('predictions:0')
        text_length = loaded_graph.get_tensor_by_name('text_length:0')
        summary_length = loaded_graph.get_tensor_by_name('summary_length:0')
        keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
        
        #Multiply by batch_size to match the model's input parameters
        answer_logits = sess.run(logits, {input_data: [text]*batch_size, 
                                        summary_length: [np.random.randint(5,8)], 
                                        text_length: [len(text)]*batch_size,
                                        keep_prob: 1.0})[0] 

# Remove the padding from the tweet
    pad = vocab_to_int["<PAD>"] 




#print('Original Text:', reviews.Text[random])
    print('Original Text:', input_sentence)
    #print('Original summary:', reviews.Summary[random])#clean_summaries[random]

    print('\nText')
    print('  Word Ids:    {}'.format([i for i in text]))
    print('  Input Words: {}'.format(" ".join([int_to_vocab[i] for i in text])))

    print('\nSummary')
    print('  Word Ids:       {}'.format([i for i in answer_logits if i != pad]))
    print('  Response Words: {}'.format(" ".join([int_to_vocab[i] for i in answer_logits if i != pad])))

    return '  Response Words: {}'.format(" ".join([int_to_vocab[i] for i in answer_logits if i != pad]))
