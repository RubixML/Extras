<?php

namespace Rubix\ML\Embedders;

use Rubix\ML\DataType;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Graph\Trees\Heap;
use Rubix\ML\NeuralNet\ActivationFunctions\Sigmoid;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Specifications\SamplesAreCompatibleWithEmbedder;
use Rubix\ML\Other\Helpers\Params;
use Rubix\ML\Other\Traits\LoggerAware;

use Tensor\Matrix;
use Tensor\Vector;

use InvalidArgumentException;
use OutOfBoundsException;

/**
 * Word2Vec
 *
 * A shallow, two-layer neural network, that produces word embeddings used in NLP models. 
 * This implementation utilizes the skip-gram algorithm and hierarchical softmax or negative sampling.
 *
 * References:
 * [1] `Tomas Mikolov et al: Efficient Estimation of Word Representations
 * in Vector Space <https://arxiv.org/pdf/1301.3781.pdf>`_, `Tomas Mikolov et al: Distributed Representations of Words
 * and Phrases and their Compositionality <https://arxiv.org/abs/1310.4546>`
 *
 * @category    Machine Learning
 * @package     RubixML
 * @author      Rich Davis
 */
class Word2Vec implements Embedder
{	
	use LoggerAware;

    /**
     * An array of sanitized and exploded sentences.
     * 
     * @var array[]     
     */	
    protected $corpus = array();

    /**
     * An array of each word in the corpus for preprocessing purposes.
     *
     * @var int[]     
     */		
	protected $raw_vocab = array();

    /**
     * An array containing each word in the corpus and it's respective index, count, and multiplier.
     *      
     * @var mixed[]
     */		
	protected $vocab = array();

    /**
     * An array containing each word in the corpus at it's respective index.
     *
     * @var int[]     
     */			
	protected $index2word = array();

    /**
     * An array containing the hidden layer.
     *
     * @var Vector[]     
     */		
	protected $syn1 = array();

    /**
     * An array containing word vectors.
     *
     * @var Vector[]     
     */		
	protected $vectors = array();	

    /**
     * The error vector used during training.
     *
     * @var \Tensor\Vector   
     */		
	protected $neu1e;	

    /**
     * The lock factors for each word in the context.
     *
     * @var int[]     
     */		
	protected $vectors_lockf = array();	

    /**
     * The L2-normalized word vectors from the model.
     *
     * @var Vector[]     
     */		
	protected $vectors_norm = array();		

    /**
     * The cumulative distrubtion table for negative sampling.
     *
     * @var int[]   
     */		
	protected $cum_table = array();	

    /**
     * The last digit in the cumulative distribution table.
     *
     * @var int   
     */		
	protected $endCumDigit;		

    /**
     * The negative labels used in the cumulative distrubtion table.
     *
     * @var \Tensor\Vector   
     */		
	protected $neg_labels;	

    /**
     * The training method determined by the layer they determine.
     *
     * @var string 
     */		
	protected $train_method;		

    /**
     * Presetting random multiplier used to validate word probabilities in sub sampling.
     *
     * @var int   
     */		

	protected $rand_multiplier = 4294967296;		
    /**
     * The layer of the network, accepts 'neg' or 'hs'.
     *
     * @var string     
     */		
	protected $layer;

    /**
     * The window size for the skip-gram model.
     *
     * @var int     
     */		
	protected $window;

    /**
     * Vector dimension size.
     *
     * @var int     
     */		
	protected $dimensions;	

    /**
     * The degree to which noise words are removed from the training text.
     *
     * @var float     
     */		
	protected $sampleRate;	

    /**
     * The amount of L2 regularization applied to the weights of the output layer.
     *
     * @var float     
     */		
	protected $alpha;	

    /**
     * The maximum number of training epochs. i.e. the number of times to iterate
     * over the entire training set before terminating.
     *
     * @var int     
     */		
	protected $epochs;	

    /**
     * The minimum times a word must appear in the corpus to be considered in the training text.
     *
     * @var int     
     */		
	protected $minCount;	

    /**
     * The total number of unique words in the corpus.
     *
     * @var int     
     */		
	protected $vocab_count;						

    /**
     * @param string $layer
     * @param int $window
     * @param int $dimensions
     * @param float $sampleRate
     * @param float $alpha
     * @param int $epochs
     * @param int $minCount
     * @throws \InvalidArgumentException
     */
	function __construct(
		string $layer     = 'neg',
		int $window       = 2,		
		int $dimensions   = 5,
		float $sampleRate = 1e-3,
		float $alpha      = 0.01,
		int $epochs       = 10,
		int $minCount     = 2
	) {	

		if(!$this->acceptedLayer($layer)){
			throw new InvalidArgumentException('Layer must be neg or hs.');
		}

		if($window > 5){
			throw new InvalidArgumentException('Window must be less than 5, $window given.');
		}

        if ($dimensions < 5) {
            throw new InvalidArgumentException("Dimensions must be greater than 4, $dimensions given.");
        }

        if ($sampleRate < 0.0) {
            throw new InvalidArgumentException("Sample rate must be 0 or greater, $sampleRate given.");
        }

        if ($alpha < 0.0) {
            throw new InvalidArgumentException("Alpha must be greater than 0, $alpha given.");
        }

        if ($epochs < 1) {
            throw new InvalidArgumentException("Number of epochs must be greater than 0, $epochs given.");
        }	

        if ($minCount < 1) {
            throw new InvalidArgumentException("Minimum word count must be greater than 0, $minCount given.");
        }	        	

		$this->layer 	    = $layer;
		$this->window 	    = $window;		
		$this->dimensions 	= $dimensions;
		$this->sampleRate   = $sampleRate;
		$this->alpha        = $alpha;		
		$this->epochs     	= $epochs;
		$this->minCount  	= $minCount;
		$this->neg_labels 	= Vector::quick([1, 0]);
	}

    /**
     * Return the data types that this estimator is compatible with.
     *
     * @return \Rubix\ML\DataType[]
     */
    public function compatibility() : array
    {
        return [
            DataType::categorical(),
        ];
    }	
	
    /**
     * Return the settings of the hyper-parameters in an associative array.
     *
     * @return mixed[]
     */
    public function params() : array
    {
        return [
            'layer'       => $this->layer,
            'window'      => $this->window,
            'dimensions'  => $this->dimensions,
            'sample_rate' => $this->sampleRate,
            'alpha'       => $this->alpha,
            'epochs'      => $this->epochs,
            'min_count'   => $this->minCount,
        ];
    }

    /**
     * Has the learner been trained?
     *
     * @return bool
     */
    public function trained() : bool
    {	
    	if(empty($this->vectors)) {
    		return false;
    	}

    	return true;
    }

    /**
     * Iterating through a range of the specific epoch count and updating all respective word vectors.  
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     *
     */		
	public function train(Dataset $dataset) : void	
	{
		DatasetIsNotEmpty::check($dataset);

		$sentences = $dataset->column(0);

		$this->preprocess($sentences);
		$this->prepareWeights();

		$min_alpha   = .0001;
		$start_alpha = $this->alpha;

		foreach(range(0, ( $this->epochs - 1)) as $i){	
			$cur_epoch       = $i;
			$this->alpha     = $start_alpha - (( $start_alpha -  $min_alpha)  * ($cur_epoch) / $this->epochs);

			$this->train_epoch_sg();						
		}

		$this->generateL2Norm();
	}

    /**
     * Scanning vocab, preparing vocab, sorting vocab, setting sampling methods
     * @param string[] $sentences
     *
     */	
	private function preprocess(array $sentences) : void
	{
		$words = $this->prepCorpus($sentences);

		$this->scanVocab($words);
		$this->prepVocab();
		$this->sortVocab();

		switch ($this->layer){
			case 'hs':
				$this->createBinaryTree();
				$this->train_method = 'train_pair_sg_hs';
				break;
			case 'neg':
				$this->createCumTable();
				$this->train_method = 'train_pair_sg_neg';
				break;
		}
	}

    /**
     * Parsing, sanitizing, & preparing array of sentences to generate & set corpus.
     *
     * @param string[] $sentences
     * @return string[] $words
     */	
	private function prepCorpus(array $sentences) : array
	{	
		$words = [];

		foreach($sentences as $sentence){		
			$prepped_sentence = $this->prepSentence($sentence);

			$this->corpus[] = $prepped_sentence;
			$words          = array_merge($words, $prepped_sentence);		
		}		

		return $words;
	}

    /**
     * Sanitizing & exploding a provided sentence.
     *
     * @param string $sentence
     * @return string[] $exploded
     */	
	private function prepSentence(string $sentence) : array
	{
		$sentence = (string)preg_replace("#[[:punct:]]#", "", strtolower($sentence));
		$sentence = (string)preg_replace('/\s\s+/', ' ', str_replace("\n", " ", $sentence));
		$sentence = trim($sentence);
	    $exploded = explode(" ", $sentence);

		return $exploded;
	}

    /**
     * Preparing and setting an initial raw vocab array for additional preprocessing
     *
     * @param string[] $words
     */	
	private function scanVocab(array $words) : void
	{
		foreach($words as $i=>$word){
		    $this->raw_vocab[$word] = (int)( $this->raw_vocab[$word] ?? 0 ) + 1;
		}
	}

    /**
     * Iterating through each word in raw vocab and formally appending word, and corresponding word count, index, and word, to vocab property.
     * Creating initial index2word property to manage word counts.
     *
     */	
	private function prepVocab() : void
	{
		$drop_total   = $drop_unique = $retain_total = 0;
		$retain_words = [];		

		foreach($this->raw_vocab as $word=>$v){
			if($v >= $this->minCount){
				$retain_words[] = $word;
				$retain_total += $v;

				$this->vocab[$word] = array('count' => $v, 'index' => count($this->index2word), 'word' => $word);
				$this->index2word[] = $word;				
			}else{
				$drop_unique += 1;
				$drop_total += $v;
			}
		}	
		
		$original_unique_total = count($retain_words) + $drop_unique;
		$retain_unique_pct     = ( ( count($retain_words) * 100 ) / $original_unique_total );

		$original_total = $retain_total + $drop_total;
		$retain_pct     = ( ( $retain_total * 100 ) / $original_total );			

		$threshold_count = $this->thresholdCount($retain_total);

		$this->updateWordProbs($retain_words, $threshold_count);
	}

    /**
     * Determine threshold word count based off of subsampling rate.
     *
     * @param int $retain_total
     * @return float $threshold_count
     */	
	private function thresholdCount(int $retain_total) : float
	{
		if(!$this->sampleRate){
			$threshold_count = $retain_total;
		}elseif($this->sampleRate < 1){
			$threshold_count = $this->sampleRate * $retain_total;
		}else{
			$threshold_count = ( $this->sampleRate * (3 + sqrt(5)) / 2 );
		}

		return $threshold_count;		
	}

    /**
     * Assigns word probabilities, determined by subsampling rate, to each word in vocabulary.
     *
     * @param string[] $retain_words
     * @param float $threshold_count
     */		
	private function updateWordProbs(array $retain_words, float $threshold_count) : void
	{
		$downsample_total  = $downsample_unique = 0;

		foreach($retain_words as $w){
			$v = $this->raw_vocab[$w];
			$word_probability = (sqrt($v / $threshold_count) + 1) * ($threshold_count / $v);

			if($word_probability < 1){
				$downsample_unique += 1;
				$downsample_total  += $word_probability * $v;
			}else{
				$word_probability = 1;
				$downsample_total += $v;
			}

			$this->vocab[$w]['sample_int'] = round($word_probability * pow(2, 32));
		}		
	}

    /**
     * Sorts vocabulary, creates index2word property, and assigns respective word index to each vocabulary word.
     *
     */	
	private function sortVocab() : void
	{
		$original = $this->vocab;

		$count = array_column($original, 'count');
		array_multisort($count, SORT_DESC, $original);

		$this->index2word = array_column($original, 'word');

		foreach($this->index2word as $index=>$word){
			$this->vocab[$word]['index'] = $index;
		}	

		$this->vocab_count = count($this->vocab);
	}

    /**
     * Creates & sets cumulative distribution table for Negative Sampling
     *
     */	
	private function createCumTable() : void
	{
		$ns_exponent     = 0.75;
		$domain 	     = (pow(2, 31) - 1);		
		$train_words_pow = $cumulative = 0;
		$cum_table       = array_fill(0, $this->vocab_count, 0);

		foreach(range(0, ($this->vocab_count - 1)) as $word_index){
			$train_words_pow += pow($this->vocab[$this->index2word[$word_index]]['count'], $ns_exponent);
		}
		
		foreach(range(0, ($this->vocab_count - 1)) as $word_index){
			$cumulative += pow($this->vocab[$this->index2word[$word_index]]['count'], $ns_exponent);
			$cum_table[$word_index] = (int)round(($cumulative / $train_words_pow) * $domain);
		}

		$this->cum_table    = $cum_table;
		$this->endCumDigit  = (int)end($cum_table);
	}

    /**
     * Creates & sets binary tree for Hierarchical Softmax Sampling
     *
     */			
	private function createBinaryTree() : void
	{
		$heap 	   = $this->buildHeap($this->vocab);		
		$max_depth = 0;
		$stack     = [[$heap[0], [], []]];
		
		while ($stack){
			$stack_item = array_pop($stack);
			
			if(empty($stack_item)) {
				break;
			}

			$points = $stack_item[2];
			$codes  = $stack_item[1];
			$node   = $stack_item[0];	

			if($node['index'] < $this->vocab_count){
				$this->vocab[$node['word']]['code']  = Vector::quick($codes);
				$this->vocab[$node['word']]['point'] = $points;

				$max_depth = max(count($codes), $max_depth);
			}else{
				$points[]  = ($node['index'] - $this->vocab_count);
				$code_left = $code_right = $codes;

				$code_left[]  = 0;
				$code_right[] = 1;

				$stack[] = array($node['left'], $code_left, $points);
				$stack[] = array($node['right'], $code_right, $points);
			}
		}
	}

    /**
     * Builds a heap queue, prioritizing each word's respective word count, to initialize the binary tree. 
     * Vocabulary array must include count index and value for each word.
     *
     * @param array[] $vocabulary
     * @return array[] $heap
     */		
	private function buildHeap(array $vocabulary) : array
	{
		$heap      = new Heap($vocabulary);
		$max_range = (count($vocabulary) - 2);

		foreach(range(0, $max_range) as $i){
			$min_1 = $heap->heappop();
			$min_2 = $heap->heappop();

			if(!empty($min_1) && !empty($min_2)){
				$new_item = array(
					'count' => ( $min_1['count'] + $min_2['count'] ),
					'index' => ( $i + (count($vocabulary)) ),
					'left'  => $min_1,
					'right' => $min_2
				);

				$heap->heappush($new_item);				
			}
		}
	
		return $heap->heap();	
	}

    /**
     * Assigning a random vector for each word in the corpus, instead of creating a massive random matrix for RAM purposes.   
     * Creating zeroed vectors for syn and neu1e.
     *
     */		
	private function prepareWeights() : void
	{
		foreach(range(0, ($this->vocab_count - 1)) as $i){
			$this->syn1[] 	   = Vector::quick(array_fill(0, $this->dimensions, 0));
			$this->vectors[$i] = Vector::rand($this->dimensions)->subtractScalar(0.5)->divideScalar($this->dimensions);
		}

		$this->neu1e         = Vector::quick(array_fill(0, $this->dimensions, 0));
		$this->vectors_lockf = array_fill(0, $this->vocab_count, 1);
	}

    /**
     * Training one epoch from the corpus and updating all respective word vectors.   
     *
     */	
	private function train_epoch_sg() : void
	{
		foreach($this->corpus as $sentence){
			$word_vocabs = $this->wordVocabs($sentence);

			foreach($word_vocabs as $pos=>$word){
				$subset = $this->sgSubset($pos, $word_vocabs);

				foreach($subset as $pos2=>$word2){
					if($pos2 !== $pos){				
						$word_index    = (string)$this->index2word[$word['index']];
						$context_index = $word2['index'];

						$this->train_pair_sg($word_index, $context_index);
					}
				}
			}
		}
	}

    /**
     * Builds an array of Word Vocabs that exceed a random multiplier from the sentence for more accurate and faster training   
     *
     * @param string[] $sentence
     * @return array[] $word_vocabs
     */	
	private function wordVocabs(array $sentence) : array
	{
		$word_vocabs = [];
		$rand        = lcg_value();
		//$rand = (rand() / getrandmax()));

		foreach($sentence as $w){
			$vocab_item = $this->vocab[$w] ?? false;

			if(!empty($vocab_item) && $vocab_item['sample_int'] > ( $rand * $this->rand_multiplier)) {
				$word_vocabs[] = $vocab_item;
			}
		}

		return $word_vocabs;
	}

    /**
     * Builds an array from the word vocab in skip-gram sequence
     *
     * @param int $pos
     * @param array[] $word_vocabs
     * @return array[] 
     */	
	private function sgSubset(int $pos, array $word_vocabs) : array
	{
		$reduced_window = rand(0, ($this->window - 1)); 
		$array_start    = max(0, ($pos - $this->window + $reduced_window));
		$array_end 	    = $pos + $this->window + 1 - $reduced_window - $array_start;

		return array_slice($word_vocabs, $array_start, $array_end, true);			
	}

    /**
     * Determining appropriate word pair training method and updating the word's vector weights.  
     *
     * @param string $word_index
     * @param int $context_index
     */	
	private function train_pair_sg(string $word_index, int $context_index) : void
	{
		$predict_word = $this->vocab[$word_index];
		$l1 		  = $this->vectors[$context_index];
		$lock_factor  = $this->vectors_lockf[$context_index];
		$train_method = $this->train_method;

		$neu1e = $this->$train_method($predict_word, $l1);

		$this->vectors[$context_index] = $l1->addVector($neu1e->multiplyScalar($lock_factor));
	}

    /**
     * Calculate the weight of the word sample using hierarchical softmax.  
     *
     * @param mixed[] $predict_word
     * @param \Tensor\Vector $l1
     * @return \Tensor\Vector
     */
	private function train_pair_sg_hs( array $predict_word, Vector $l1) : Vector
	{
		$word_indices = $predict_word['point'];

		$l2 = $this->layerMatrix($word_indices);
		$fa = $this->propagateHidden($l2, $l1);
		$gb = $fa->addVector($predict_word['code'])->negate()->addScalar(1)->multiplyScalar($this->alpha);

		$this->learnHidden($word_indices, $gb, $l1);

		return $this->neu1e->addMatrix($gb->matmul($l2))->rowAsVector(0);
	}

    /**
     * Calculate & return new vector weight of word sample using negative sampling.  
     *
     * @param array[] $predict_word
     * @param \Tensor\Vector $l1
     * @return \Tensor\Vector
     */
	private function train_pair_sg_neg(array $predict_word, Vector $l1) : Vector
	{		
		$word_indices = [$predict_word['index']];

		while(count($word_indices) < 1 + 1){
			$temp     = $this->cum_table;			
			$rand_int = rand(0, $this->endCumDigit);					
			$temp[]   = $rand_int;

			sort($temp);
			$w = array_search($rand_int, $temp);

			if($w !== $predict_word['index']){
				$word_indices[] = $w;
			}

			continue;
		}			
		
		$l2 = $this->layerMatrix($word_indices);
		$fa = $this->propagateHidden($l2, $l1);
		$gb = $this->neg_labels->subtractVector($fa)->multiplyScalar($this->alpha);

		$this->learnHidden($word_indices, $gb, $l1);

		return $this->neu1e->addMatrix($gb->matmul($l2))->rowAsVector(0);
	}

    /**
     * Creating 2-d matrix from word indices.
     *
     * @param mixed[] $word_indices
     * @return \Tensor\Matrix 
     */		
	private function layerMatrix(array $word_indices) : Matrix
	{
		$l2a = array();

		foreach($word_indices as $index){
			$l2a[] = $this->syn1[$index];			
		}		

		return Matrix::stack($l2a);
	}

    /**
     * Updating the hidden layer for given word index supplied the outer product of the word vector and error gradients.
     *
     * @param mixed[] $word_indices
     * @param \Tensor\Vector $g
     * @param \Tensor\Vector $l1
     */	
	private function learnHidden( array $word_indices,  Vector $g, Vector $l1) : void
	{
		$c     = $g->outer($l1);
		$count = 0;

		foreach($word_indices as $index){
			$this->syn1[$index] = $this->syn1[$index]->addVector($c->rowAsVector($count));
			$count += 1;
		}			
	}	

    /**
     * Propagating hidden layer.
     *
     * @param \Tensor\Matrix $l2
     * @param \Tensor\Vector $l1
     * @return \Tensor\Vector
     */	
	private function propagateHidden(Matrix $l2, Vector $l1) : Vector
	{
		$prod_term = $l1->matmul($l2->transpose());
		$sigmoid   = new Sigmoid();

		return $sigmoid->compute($prod_term)->rowAsVector(0);
	}

    /**
     * Generating L2 Norm of final word vectors. 
     *
     */	
	private function generateL2Norm() : void
	{	
		$l2_norm = [];

		foreach($this->vectors as $vector){	
			$l2_norm[] = $vector->divideScalar($vector->L2Norm());
		}	

		$this->vectors_norm = $l2_norm;
	}	

    /**
     * Embed a high dimensional dataset into a lower dimensional one.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     * @return array[] $embeddedDataset
     */
    public function embed(Dataset $dataset) : array
    {
        DatasetIsNotEmpty::check($dataset);
        SamplesAreCompatibleWithEmbedder::check($dataset, $this); 

        if ($this->logger) {
            $this->logger->info('Embedder init ' . Params::stringify($this->params()));

            $this->logger->info('Computing high-dimensional affinities');
        }

        $samples         = $dataset->samples();
        $embeddedDataset = [];

		foreach($samples as $i => $featureSet){
			foreach($featureSet as $i2 => $sentence){
				$preppedSentence = $this->prepSentence($sentence);

				$embeddings = [];
				foreach($preppedSentence as $word){
					$embeddings[] = $this->embedWord($word);
				}

				$embeddedDataset[][$i2] = Matrix::stack($embeddings)->transpose()->mean();
			}
		}

		return $embeddedDataset;
    }

    /**
     * Determining the top N similar words provided an array of positive words and negative words.
     *
     * @param string[] $positive
     * @param string[] $negative
     * @param int $top
     * @return string[] $result
     */	
	public function most_similar(array $positive, array $negative = array(), $top = 20) : array
	{
		$positive_array = $negative_array = $all_words = $means = array();

		foreach($positive as $word){
			$positive_array[$word] = 1.0;
		}
		
		foreach($negative as $word){
			$negative_array[$word] = -1.0;
		}				

		$word_array = array_merge($positive_array, $negative_array);

		foreach($word_array as $word=>$weight){
			$wordEmbedding = $this->wordVec($word);

			if($wordEmbedding instanceof Vector){
				$means[] 	   = $wordEmbedding->multiplyScalar($weight);
				$all_words[]   = $this->vocab[$word]['index'];				
			}
		}

		if(empty($all_words)){
			throw new InvalidArgumentException('Positive words were not found in vocab.');
		}

		$mean  = Matrix::stack($means)->transpose()->mean();
		$l2    = Matrix::stack($this->vectors_norm);
		$dists = $mean->transpose()->matmul($l2->transpose())->asArray()[0];

		arsort($dists, 1);		

		$result = array();
		foreach($dists as $index=>$weight){
			if(!in_array($index, $all_words)) {
				$result[$this->index2word[$index]] = $weight;
			}
		}	

		$result = array_slice($result, 0, $top, true);	

		return $result;
	}	

    /**
     * Returns the word embedding for a given word.
     *
     * @param string $word
     * @param bool $use_norm
     * @return \Tensor\Vector|null $result
     */	
	public function wordVec(string $word, bool $use_norm = True) : ?Vector
	{
		if(!array_key_exists($word, $this->vocab)){
			return null;
		}

		if($use_norm){
			$result = $this->vectors_norm[$this->vocab[$word]['index']];
		}else{
			$result = $this->vectors[$this->vocab[$word]['index']];
		}

		return $result;
	}

    /**
     * Returns the word embedding, or a vector of zeros if empty, for a given word.
     *
     * @param string $word
     * @param bool $use_norm     
     * @return \Tensor\Vector $result
     */	
	public function embedWord(string $word, bool $use_norm = True) : Vector
	{
		$wordEmbedding = $this->wordVec($word);

		if(!$wordEmbedding){
			$wordEmbedding = Vector::quick(array_fill(0, $this->dimensions, 0));
		}

		return $wordEmbedding;
	}

    /**
     * Determining if layer is acceptable.
     *
     * @param string $layer
     * @return bool
     */
    private function acceptedLayer(string $layer) : bool
    {
    	$accepted_layers = array('hs', 'neg');

    	if(!in_array($layer, $accepted_layers)){
    		return false;
    	}

    	return true;
    }


}