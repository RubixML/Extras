<?php

namespace Rubix\ML\Persisters;

use League\Flysystem\FilesystemInterface;
use Rubix\ML\Encoding;
use Rubix\ML\Helpers\Params;
use Rubix\ML\Exceptions\RuntimeException;

if (interface_exists('League\Flysystem\FilesystemInterface')) {
    /**
     * FlysystemV1
     *
     * Flysystem is a filesystem library providing a unified storage interface and abstraction layer.
     * It enables access to many different storage backends such as Local, Amazon S3, FTP, and more.
     *
     * > **Note:** The FlysystemV1 persister is designed to work with Flysystem version 1.x
     *
     * @see https://flysystem.thephpleague.com
     *
     * @category    Machine Learning
     * @package     Rubix/ML
     * @author      Chris Simpson
     */
    class FlysystemV1 implements Persister
    {
        /**
         * The extension to give files created as part of a persistable's save history.
         *
         * @var string
         */
        public const HISTORY_EXT = 'old';

        /**
         * The path to the model file on the filesystem.
         *
         * @var string
         */
        protected $path;

        /**
         * The filesystem implementation providing access to your backend storage.
         *
         * @var \League\Flysystem\FilesystemInterface
         */
        protected $filesystem;

        /**
         * Should we keep a history of past saves?
         *
         * @var bool
         */
        protected $history;

        /**
         * @param string $path
         * @param FilesystemInterface $filesystem
         * @param bool $history
         */
        public function __construct(string $path, FilesystemInterface $filesystem, bool $history = false)
        {
            $this->path = $path;
            $this->filesystem = $filesystem;
            $this->history = $history;
        }

        /**
         * Save an encoding.
         *
         * @param \Rubix\ML\Encoding $encoding
         *
         * @throws \Rubix\ML\Exceptions\RuntimeException
         */
        public function save(Encoding $encoding) : void
        {
            if ($this->history and $this->filesystem->has($this->path)) {
                $timestamp = (string) time();

                $filename = $this->path . '-' . $timestamp . '.' . self::HISTORY_EXT;

                $num = 0;

                while ($this->filesystem->has($filename)) {
                    $filename = $this->path . '-' . $timestamp . '-' . ++$num . '.' . self::HISTORY_EXT;
                }

                try {
                    if (!$this->filesystem->rename($this->path, $filename)) {
                        throw new RuntimeException("Failed to create history file: '{$filename}' {$this}");
                    }
                } catch (League\Flysystem\Exception $e) {
                    throw new RuntimeException("Failed to create history file: '{$filename}' {$this}");
                }
            }

            $success = $this->filesystem->put($this->path, (string) $encoding);

            if (!$success) {
                throw new RuntimeException("Could not write to filesystem. {$this}");
            }
        }

        /**
         * Load the last model that was saved.
         *
         * @throws \RuntimeException
         * @return \Rubix\ML\Persistable
         */
        public function load() : Encoding
        {
            if (!$this->filesystem->has($this->path)) {
                throw new RuntimeException("File does not exist at {$this->path}.");
            }

            $encoding = new Encoding($this->filesystem->read($this->path) ?: '');

            if ($encoding->bytes() === 0) {
                throw new RuntimeException("File at {$this->path} does not contain any data.");
            }

            return $encoding;
        }

        /**
         * Return the string representation of the object.
         *
         * @return string
         */
        public function __toString() : string
        {
            return "Flysystem (path: {$this->path}, filesystem: "
                . Params::toString($this->filesystem) . ', history: '
                . Params::toString($this->history) . ')';
        }
    }
}
