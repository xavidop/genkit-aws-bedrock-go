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

// Package main demonstrates basic usage of the AWS Bedrock plugin
package main

import (
	"context"
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

	log.Println("Starting basic Bedrock example...")

	// Define Claude 3 Haiku model
	claudeModel := bedrockPlugin.DefineModel(g, bedrock.ModelDefinition{
		Name: "anthropic.claude-3-haiku-20240307-v1:0",
		Type: "chat",
	}, nil)

	log.Printf("Defined model: %v", claudeModel)

	// Example: Generate text (basic usage)
	response, err := genkit.Generate(ctx, g,
		ai.WithModel(claudeModel),
		ai.WithPrompt("What are the key benefits of using AWS Bedrock for AI applications?"),
	)
	if err != nil {
		log.Printf("Error generating text: %v", err)
	} else {
		log.Printf("Generated response: %s", response.Text())
	}

	log.Println("Basic Bedrock example completed")
}
