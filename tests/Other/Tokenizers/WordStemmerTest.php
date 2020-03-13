<?php

namespace Rubix\ML\Tests\Other\Tokenizers;

use Rubix\ML\Other\Tokenizers\Tokenizer;
use Rubix\ML\Other\Tokenizers\WordStemmer;
use PHPUnit\Framework\TestCase;

/**
 * @group Tokenizers
 * @covers \Rubix\ML\Other\Tokenizers\WordStemmer
 */
class WordStemmerTest extends TestCase
{
    /**
     * @var \Rubix\ML\Other\Tokenizers\Word
     */
    protected $tokenizer;
    
    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->tokenizer = new WordStemmer('english');
    }
    
    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(WordStemmer::class, $this->tokenizer);
        $this->assertInstanceOf(Tokenizer::class, $this->tokenizer);
    }
    
    /**
     * @test
     */
    public function tokenize() : void
    {
        $text = 'I would like to die on Mars, just not on impact. Majority voting.';

        $expected = [
            'I', 'would', 'like', 'to', 'die', 'on', 'mar', 'just', 'not', 'on', 'impact',
            'major', 'vote',
        ];

        $tokens = $this->tokenizer->tokenize($text);

        $this->assertCount(13, $tokens);
        $this->assertEquals($expected, $tokens);
    }
}
