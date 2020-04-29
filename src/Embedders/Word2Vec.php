<?php

namespace Rubix\ML\Embedders;

use Rubix\ML\DataType;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Graph\Trees\Heap;
use Rubix\ML\NeuralNet\ActivationFunctions\Sigmoid;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Specifications\SamplesAreCompatibleWithEmbedder;
use Rubix\ML\Other\Helpers\Params;
use Tensor\Matrix;
use Tensor\Vector;
use InvalidArgumentException;

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

    /**
     * An array of sanitized and exploded sentences.
     *
     * @var array[]
     */
    protected $corpus = [];

    /**
     * An array of each word in the corpus for preprocessing purposes.
     *
     * @var int[]
     */
    protected $rawVocab = [];

    /**
     * An array containing each word in the corpus and it's respective index, count, and multiplier.
     *
     * @var mixed[]
     */
    protected $vocab = [];

    /**
     * An array containing each word in the corpus at it's respective index.
     *
     * @var int[]
     */
    protected $index2word = [];

    /**
     * An array containing the hidden layer.
     *
     * @var Vector[]
     */
    protected $syn1 = [];

    /**
     * An array containing word vectors.
     *
     * @var Vector[]
     */
    protected $vectors = [];

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
    protected $vectorsLockf = [];

    /**
     * The L2-normalized word vectors from the model.
     *
     * @var Vector[]
     */
    protected $vectorsNorm = [];

    /**
     * The cumulative distrubtion table for negative sampling.
     *
     * @var int[]
     */
    protected $cumTable = [];

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
    protected $negLabels;

    /**
     * The training method determined by the layer they determine.
     *
     * @var string
     */
    protected $trainMethod;

    /**
     * Presetting random multiplier used to validate word probabilities in sub sampling.
     *
     * @var int
     */
    protected $randMultiplier = 4294967296;

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
    protected $vocabCount;

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
    public function __construct(
        string $layer = 'neg',
        int $window = 2,
        int $dimensions = 5,
        float $sampleRate = 1e-3,
        float $alpha = 0.01,
        int $epochs = 10,
        int $minCount = 2
    ) {
        if(!in_array($layer, ['neg', 'hs'])) {
            throw new InvalidArgumentException("Layer must be neg or hs.");
        }

        if ($window > 5) {
            throw new InvalidArgumentException("Window must be between 1 and 5, $window given.");
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

        $this->layer = $layer;
        $this->window = $window;
        $this->dimensions = $dimensions;
        $this->sampleRate = $sampleRate;
        $this->alpha = $alpha;
        $this->epochs = $epochs;
        $this->minCount = $minCount;
        $this->negLabels = Vector::quick([1, 0]);
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
            'layer' => $this->layer,
            'window' => $this->window,
            'dimensions' => $this->dimensions,
            'sample_rate' => $this->sampleRate,
            'alpha' => $this->alpha,
            'epochs' => $this->epochs,
            'min_count' => $this->minCount,
        ];
    }

    /**
     * Has the learner been trained?
     *
     * @return bool
     */
    public function trained() : bool
    {
        return !empty($this->vectors);
    }

    /**
     * Iterating through a range of the specific epoch count and updating all respective word vectors.
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     */
    public function train(Dataset $dataset) : void
    {
        DatasetIsNotEmpty::check($dataset);

        $sentences = $dataset->column(0);

        $this->preprocess($sentences);
        $this->prepareWeights();

        $minAlpha = 0.0001;
        $startAlpha = $this->alpha;

        for ($i = 0; $i < $this->epochs; ++$i) {
            $this->alpha = $startAlpha - (($startAlpha - $minAlpha) * ($i) / $this->epochs);

            $this->train_epoch_sg();
        }

        $this->generateL2Norm();
    }

    /**
     * Scanning vocab, preparing vocab, sorting vocab, setting sampling methods
     *
     * @param string[] $sentences
     */
    private function preprocess(array $sentences) : void
    {
        $words = $this->prepCorpus($sentences);

        $this->scanVocab($words);
        $this->prepVocab();
        $this->sortVocab();

        switch ($this->layer) {
            case 'hs':
                $this->createBinaryTree();
                $this->trainMethod = 'train_pair_sg_hs';
                break;
            case 'neg':
                $this->createCumTable();
                $this->trainMethod = 'train_pair_sg_neg';
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

        foreach ($sentences as $sentence) {
            $preppedSentence = $this->prepSentence($sentence);

            $this->corpus[] = $preppedSentence;
            $words = array_merge($words, $preppedSentence);
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
        $sentence = (string) preg_replace('#[[:punct:]]#', '', strtolower($sentence));
        $sentence = (string) preg_replace('/\s\s+/', ' ', str_replace("\n", ' ', $sentence));
        $sentence = trim($sentence);
        $exploded = explode(' ', $sentence);

        return $exploded;
    }

    /**
     * Preparing and setting an initial raw vocab array for additional preprocessing
     *
     * @param string[] $words
     */
    private function scanVocab(array $words) : void
    {
        foreach ($words as $i => $word) {
            $this->rawVocab[$word] = (int) ($this->rawVocab[$word] ?? 0) + 1;
        }
    }

    /**
     * Iterating through each word in raw vocab and formally appending word, and corresponding word count, index, and word, to vocab property.
     * Creating initial index2word property to manage word counts.
     */
    private function prepVocab() : void
    {
        $dropTotal = $dropUnique = $retainTotal = 0;
        $retainWords = [];

        foreach ($this->rawVocab as $word => $v) {
            if ($v >= $this->minCount) {
                $retainWords[] = $word;
                $retainTotal += $v;

                $this->vocab[$word] = ['count' => $v, 'index' => count($this->index2word), 'word' => $word];
                $this->index2word[] = $word;
            } else {
                $dropUnique += 1;
                $dropTotal += $v;
            }
        }
        
        $originalUniqueTotal = count($retainWords) + $dropUnique;
        $retainUniquePct = ((count($retainWords) * 100) / $originalUniqueTotal);

        $originalTotal = $retainTotal + $dropTotal;
        $retainPct = (($retainTotal * 100) / $originalTotal);

        $thresholdCount = $this->thresholdCount($retainTotal);

        $this->updateWordProbs($retainWords, $thresholdCount);
    }

    /**
     * Determine threshold word count based off of subsampling rate.
     *
     * @param int $retainTotal
     * @return float $thresholdCount
     */
    private function thresholdCount(int $retainTotal) : float
    {
        if (!$this->sampleRate) {
            $thresholdCount = $retainTotal;
        } elseif ($this->sampleRate < 1) {
            $thresholdCount = $this->sampleRate * $retainTotal;
        } else {
            $thresholdCount = ($this->sampleRate * (3 + sqrt(5)) / 2);
        }

        return $thresholdCount;
    }

    /**
     * Assigns word probabilities, determined by subsampling rate, to each word in vocabulary.
     *
     * @param string[] $retainWords
     * @param float $thresholdCount
     */
    private function updateWordProbs(array $retainWords, float $thresholdCount) : void
    {
        $downsampleTotal = $downsampleUnique = 0;

        foreach ($retainWords as $w) {
            $v = $this->rawVocab[$w];
            $wordProbability = (sqrt($v / $thresholdCount) + 1) * ($thresholdCount / $v);

            if ($wordProbability < 1) {
                $downsampleUnique += 1;
                $downsampleTotal += $wordProbability * $v;
            } else {
                $wordProbability = 1;
                $downsampleTotal += $v;
            }

            $this->vocab[$w]['sample_int'] = round($wordProbability * pow(2, 32));
        }
    }

    /**
     * Sorts vocabulary, creates index2word property, and assigns respective word index to each vocabulary word.
     */
    private function sortVocab() : void
    {
        $original = $this->vocab;

        $count = array_column($original, 'count');
        array_multisort($count, SORT_DESC, $original);

        $this->index2word = array_column($original, 'word');

        foreach ($this->index2word as $index => $word) {
            $this->vocab[$word]['index'] = $index;
        }

        $this->vocabCount = count($this->vocab);
    }

    /**
     * Creates & sets cumulative distribution table for Negative Sampling
     */
    private function createCumTable() : void
    {
        $nsExponent = 0.75;
        $domain = (pow(2, 31) - 1);
        $trainWordsPow = $cumulative = 0;
        $cumTable = array_fill(0, $this->vocabCount, 0);

        foreach (range(0, ($this->vocabCount - 1)) as $wordIndex) {
            $trainWordsPow += pow($this->vocab[$this->index2word[$wordIndex]]['count'], $nsExponent);
        }
        
        foreach (range(0, ($this->vocabCount - 1)) as $wordIndex) {
            $cumulative += pow($this->vocab[$this->index2word[$wordIndex]]['count'], $nsExponent);
            $cumTable[$wordIndex] = (int) round(($cumulative / $trainWordsPow) * $domain);
        }

        $this->cumTable = $cumTable;
        $this->endCumDigit = (int) end($cumTable);
    }

    /**
     * Creates & sets binary tree for Hierarchical Softmax Sampling
     */
    private function createBinaryTree() : void
    {
        $heap = $this->buildHeap($this->vocab);
        $maxDepth = 0;
        $stack = [[$heap[0], [], []]];
        
        while ($stack) {
            $stackItem = array_pop($stack);
            
            if (empty($stackItem)) {
                break;
            }

            $points = $stackItem[2];
            $codes = $stackItem[1];
            $node = $stackItem[0];

            if ($node['index'] < $this->vocabCount) {
                $this->vocab[$node['word']]['code'] = Vector::quick($codes);
                $this->vocab[$node['word']]['point'] = $points;

                $maxDepth = max(count($codes), $maxDepth);
            } else {
                $points[] = ($node['index'] - $this->vocabCount);
                $codeLeft = $codeRight = $codes;

                $codeLeft[] = 0;
                $codeRight[] = 1;

                $stack[] = [$node['left'], $codeLeft, $points];
                $stack[] = [$node['right'], $codeRight, $points];
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
        $heap = new Heap($vocabulary);
        $maxRange = (count($vocabulary) - 2);

        foreach (range(0, $maxRange) as $i) {
            $min1 = $heap->heappop();
            $min2 = $heap->heappop();

            if (!empty($min1) && !empty($min2)) {
                $new_item = [
                    'count' => ($min1['count'] + $min2['count']),
                    'index' => ($i + (count($vocabulary))),
                    'left' => $min1,
                    'right' => $min2
                ];

                $heap->heappush($new_item);
            }
        }
    
        return $heap->heap();
    }

    /**
     * Assigning a random vector for each word in the corpus, instead of creating a massive random matrix for RAM purposes.
     * Creating zeroed vectors for syn and neu1e.
     */
    private function prepareWeights() : void
    {
        foreach (range(0, ($this->vocabCount - 1)) as $i) {
            $this->syn1[] = Vector::quick(array_fill(0, $this->dimensions, 0));
            $this->vectors[$i] = Vector::rand($this->dimensions)->subtractScalar(0.5)->divideScalar($this->dimensions);
        }

        $this->neu1e = Vector::quick(array_fill(0, $this->dimensions, 0));
        $this->vectorsLockf = array_fill(0, $this->vocabCount, 1);
    }

    /**
     * Training one epoch from the corpus and updating all respective word vectors.
     */
    private function train_epoch_sg() : void
    {
        foreach ($this->corpus as $sentence) {
            $wordVocabs = $this->wordVocabs($sentence);

            foreach ($wordVocabs as $pos => $word) {
                $subset = $this->sgSubset($pos, $wordVocabs);

                foreach ($subset as $pos2 => $word2) {
                    if ($pos2 !== $pos) {
                        $wordIndex = (string) $this->index2word[$word['index']];
                        $contextIndex = $word2['index'];

                        $this->train_pair_sg($wordIndex, $contextIndex);
                    }
                }
            }
        }
    }

    /**
     * Builds an array of Word Vocabs that exceed a random multiplier from the sentence for more accurate and faster training
     *
     * @param string[] $sentence
     * @return array[] $wordVocabs
     */
    private function wordVocabs(array $sentence) : array
    {
        $wordVocabs = [];
        $rand = lcg_value();

        foreach ($sentence as $w) {
            $vocabItem = $this->vocab[$w] ?? false;

            if (!empty($vocabItem) && $vocabItem['sample_int'] > ($rand * $this->randMultiplier)) {
                $wordVocabs[] = $vocabItem;
            }
        }

        return $wordVocabs;
    }

    /**
     * Builds an array from the word vocab in skip-gram sequence
     *
     * @param int $pos
     * @param array[] $wordVocabs
     * @return array[]
     */
    private function sgSubset(int $pos, array $wordVocabs) : array
    {
        $reducedWindow = rand(0, ($this->window - 1));
        $arrayStart = max(0, ($pos - $this->window + $reducedWindow));
        $arrayEnd = $pos + $this->window + 1 - $reducedWindow - $arrayStart;

        return array_slice($wordVocabs, $arrayStart, $arrayEnd, true);
    }

    /**
     * Determining appropriate word pair training method and updating the word's vector weights.
     *
     * @param string $wordIndex
     * @param int $contextIndex
     */
    private function train_pair_sg(string $wordIndex, int $contextIndex) : void
    {
        $predictWord = $this->vocab[$wordIndex];
        $l1 = $this->vectors[$contextIndex];
        $lockFactor = $this->vectorsLockf[$contextIndex];
        $trainMethod = $this->trainMethod;

        $neu1e = $this->$trainMethod($predictWord, $l1);

        $this->vectors[$contextIndex] = $l1->addVector($neu1e->multiplyScalar($lockFactor));
    }

    /**
     * Calculate the weight of the word sample using hierarchical softmax.
     *
     * @param mixed[] $predictWord
     * @param \Tensor\Vector $l1
     * @return \Tensor\Vector
     */
    private function train_pair_sg_hs(array $predictWord, Vector $l1) : Vector
    {
        $word_indices = $predictWord['point'];

        $l2 = $this->layerMatrix($word_indices);
        $fa = $this->propagateHidden($l2, $l1);
        $gb = $fa->addVector($predictWord['code'])->negate()->addScalar(1)->multiplyScalar($this->alpha);

        $this->learnHidden($word_indices, $gb, $l1);

        return $this->neu1e->addMatrix($gb->matmul($l2))->rowAsVector(0);
    }

    /**
     * Calculate & return new vector weight of word sample using negative sampling.
     *
     * @param array[] $predictWord
     * @param \Tensor\Vector $l1
     * @return \Tensor\Vector
     */
    private function train_pair_sg_neg(array $predictWord, Vector $l1) : Vector
    {
        $wordIndices = [$predictWord['index']];

        while (count($wordIndices) < 1 + 1) {
            $temp = $this->cumTable;
            $randInt = rand(0, $this->endCumDigit);
            $temp[] = $randInt;

            sort($temp);
            $w = array_search($randInt, $temp);

            if ($w !== $predictWord['index']) {
                $wordIndices[] = $w;
            }

            continue;
        }
        
        $l2 = $this->layerMatrix($wordIndices);
        $fa = $this->propagateHidden($l2, $l1);
        $gb = $this->negLabels->subtractVector($fa)->multiplyScalar($this->alpha);

        $this->learnHidden($wordIndices, $gb, $l1);

        return $this->neu1e->addMatrix($gb->matmul($l2))->rowAsVector(0);
    }

    /**
     * Creating 2-d matrix from word indices.
     *
     * @param mixed[] $wordIndices
     * @return \Tensor\Matrix
     */
    private function layerMatrix(array $wordIndices) : Matrix
    {
        $l2a = [];

        foreach ($wordIndices as $index) {
            $l2a[] = $this->syn1[$index];
        }

        return Matrix::stack($l2a);
    }

    /**
     * Updating the hidden layer for given word index supplied the outer product of the word vector and error gradients.
     *
     * @param mixed[] $wordIndices
     * @param \Tensor\Vector $g
     * @param \Tensor\Vector $l1
     */
    private function learnHidden(array $wordIndices, Vector $g, Vector $l1) : void
    {
        $c = $g->outer($l1);
        $count = 0;

        foreach ($wordIndices as $index) {
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
        $prodTerm = $l1->matmul($l2->transpose());
        $sigmoid = new Sigmoid();

        return $sigmoid->compute($prodTerm)->rowAsVector(0);
    }

    /**
     * Generating L2 Norm of final word vectors.
     */
    private function generateL2Norm() : void
    {
        $l2Norm = [];

        foreach ($this->vectors as $vector) {
            $l2Norm[] = $vector->divideScalar($vector->L2Norm());
        }

        $this->vectorsNorm = $l2Norm;
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

        $samples = $dataset->samples();
        $embeddedDataset = [];

        foreach ($samples as $i => $featureSet) {
            foreach ($featureSet as $i2 => $sentence) {
                $preppedSentence = $this->prepSentence($sentence);

                $embeddings = [];
                foreach ($preppedSentence as $word) {
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
    public function most_similar(array $positive, array $negative = [], $top = 20) : array
    {
        $positiveArray = $negativeArray = $allWords = $means = [];

        foreach ($positive as $word) {
            $positiveArray[$word] = 1.0;
        }
        
        foreach ($negative as $word) {
            $negativeArray[$word] = -1.0;
        }

        $wordArray = array_merge($positiveArray, $negativeArray);

        foreach ($wordArray as $word => $weight) {
            $wordEmbedding = $this->wordVec($word);

            if ($wordEmbedding instanceof Vector) {
                $means[] = $wordEmbedding->multiplyScalar($weight);
                $allWords[] = $this->vocab[$word]['index'];
            }
        }

        if (empty($allWords)) {
            throw new InvalidArgumentException('Positive words were not found in vocab.');
        }

        $mean = Matrix::stack($means)->transpose()->mean();
        $l2 = Matrix::stack($this->vectorsNorm);
        $dists = $mean->transpose()->matmul($l2->transpose())->asArray()[0];

        arsort($dists, 1);

        $result = [];
        foreach ($dists as $index => $weight) {
            if (!in_array($index, $allWords)) {
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
    public function wordVec(string $word, bool $use_norm = true) : ?Vector
    {
        if (!array_key_exists($word, $this->vocab)) {
            return null;
        }

        if ($use_norm) {
            $result = $this->vectorsNorm[$this->vocab[$word]['index']];
        } else {
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
    public function embedWord(string $word, bool $use_norm = true) : Vector
    {
        $wordEmbedding = $this->wordVec($word);

        if (!$wordEmbedding) {
            $wordEmbedding = Vector::zeros(0, $this->dimensions);
        }

        return $wordEmbedding;
    }

}
