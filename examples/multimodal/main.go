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

// Package main demonstrates multimodal conversation with AWS Bedrock
package main

import (
	"context"
	"encoding/base64"
	"log"
	"os"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	bedrock "github.com/xavidop/genkit-aws-bedrock-go"
)

func main() {
	ctx := context.Background()

	bedrockPlugin := &bedrock.Bedrock{
		Region: "us-east-1",
	}

	// Initialize Genkit
	g := genkit.Init(ctx,
		genkit.WithPlugins(bedrockPlugin),
	)

	log.Println("Starting multimodal conversation example...")

	// Define Claude 3 model (supports multimodal)
	claudeModel := bedrockPlugin.DefineModel(g, bedrock.ModelDefinition{
		Name: "anthropic.claude-3-sonnet-20240229-v1:0",
		Type: "chat",
	}, nil)

	// Example conversation with both text and image
	// Read the cat.jpeg file
	imageBytes, err := os.ReadFile("cat.jpeg")
	if err != nil {
		log.Fatalf("Failed to read cat.jpeg: %v\nPlease add a cat.jpeg file to this directory to test multimodal functionality.", err)
	}

	// Convert to base64 data URL
	imageData := "data:image/jpeg;base64," + base64.StdEncoding.EncodeToString(imageBytes)

	response, err := genkit.Generate(ctx, g,
		ai.WithModel(claudeModel),
		ai.WithMessages(ai.NewUserMessage(
			ai.NewTextPart("What do you see in this image? Please describe it in detail."),
			ai.NewMediaPart("image/jpeg", imageData),
		)),
	)

	if err != nil {
		log.Printf("Error: %v", err)
		return
	}

	log.Printf("Response: %s", response.Text())
	log.Println("Multimodal conversation example completed")

}
