// Copyright 2025 Xavier Portilla Edo
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

// Package main demonstrates streaming text generation with AWS Bedrock
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	bedrock "github.com/xavidop/genkit-aws-bedrock-go"
)

func main() {
	ctx := context.Background()

	// Initialize Genkit
	g, err := genkit.Init(ctx)
	if err != nil {
		log.Fatal(err)
	}

	// Initialize Bedrock plugin
	bedrockPlugin := &bedrock.Bedrock{
		Region: "us-east-1",
	}

	if err := bedrockPlugin.Init(ctx, g); err != nil {
		log.Fatal(err)
	}

	log.Println("Starting streaming text generation example...")

	// Define Claude 3 Sonnet model
	claudeModel := bedrockPlugin.DefineModel(g, bedrock.ModelDefinition{
		Name: "anthropic.claude-3-sonnet-20240229-v1:0",
		Type: "chat",
	}, nil)

	// Streaming callback to handle response chunks
	streamCallback := func(ctx context.Context, chunk *ai.ModelResponseChunk) error {
		if len(chunk.Content) > 0 {
			for _, part := range chunk.Content {
				if part.IsText() {
					fmt.Print(part.Text)
				}
			}
		}
		return nil
	}

	log.Println("Generating streaming response...")

	// Generate streaming response
	response, err := genkit.Generate(ctx, g,
		ai.WithModel(claudeModel),
		ai.WithPrompt("Write a short story about a robot learning to paint. Make it creative and engaging."),
		ai.WithStreaming(streamCallback),
	)

	if err != nil {
		log.Printf("Error in streaming generation: %v", err)
	} else {
		fmt.Println() // New line after streaming
		log.Printf("Streaming generation completed. Final response length: %d characters", len(response.Text()))
	}

	log.Println("Streaming example completed")
}
