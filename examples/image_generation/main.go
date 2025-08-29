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

// Package main demonstrates image generation with AWS Bedrock
package main

import (
	"context"
	"encoding/base64"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	bedrock "github.com/xavidop/genkit-aws-bedrock-go"
)

// saveImageFromDataURL extracts base64 data from a data URL and saves it as an image file
func saveImageFromDataURL(dataURL, filename string) error {
	// Show a preview of the data URL (first 50 characters)
	preview := dataURL
	if len(preview) > 50 {
		preview = preview[:50] + "..."
	}
	log.Printf("Processing image data URL: %s", preview)

	// Parse the data URL to extract the base64 data
	var base64Data string
	if strings.HasPrefix(dataURL, "data:") {
		// Split by comma to separate metadata from data
		parts := strings.Split(dataURL, ",")
		if len(parts) == 2 {
			base64Data = parts[1] // The actual base64 data
		} else {
			return fmt.Errorf("invalid data URL format")
		}
	} else {
		// If it's already just base64 data without the data URL prefix
		base64Data = dataURL
	}

	// Decode base64 image data
	imgBytes, err := base64.StdEncoding.DecodeString(base64Data)
	if err != nil {
		return fmt.Errorf("failed to decode base64 data: %w", err)
	}

	// Save to file
	err = os.WriteFile(filename, imgBytes, 0644)
	if err != nil {
		return fmt.Errorf("failed to write image file: %w", err)
	}

	log.Printf("Image saved as %s (size: %d bytes)", filename, len(imgBytes))
	return nil
}

func main() {
	ctx := context.Background()

	bedrockPlugin := &bedrock.Bedrock{
		Region: "us-east-1",
	}

	// Initialize Genkit
	g := genkit.Init(ctx,
		genkit.WithPlugins(bedrockPlugin),
	)

	log.Println("Genkit initialized")

	log.Println("Starting image generation example...")

	// Define Titan Image Generator model
	titanImageModel := bedrockPlugin.DefineModel(g, bedrock.ModelDefinition{
		Name: "amazon.titan-image-generator-v1",
		Type: "image",
	}, nil)

	// Example image generation request
	prompt := "A serene mountain landscape at sunset, painted in watercolor style"

	response, err := genkit.Generate(ctx, g,
		ai.WithModel(titanImageModel),
		ai.WithPrompt(prompt),
		ai.WithConfig(map[string]interface{}{
			"numberOfImages": 1,
			"height":         1024,
			"width":          1024,
			"cfgScale":       8,
		}),
	)

	if err != nil {
		log.Printf("Error generating image: %v", err)
		return
	}

	// Save generated image
	if response != nil && response.Message != nil && len(response.Message.Content) > 0 {
		for i, part := range response.Message.Content {
			if part.IsMedia() {
				filename := fmt.Sprintf("generated_image_%d.png", i+1)
				err := saveImageFromDataURL(part.Text, filename)
				if err != nil {
					log.Printf("Failed to save image %d: %v", i+1, err)
				}
			}
		}
	} else {
		log.Println("No image content received in response")
	}

	log.Printf("Image generation example completed for prompt: %s", prompt)
}
