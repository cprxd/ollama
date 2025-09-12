package parser

import (
	"encoding/json"
	"errors"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/harmony"
)

type TokenParserType int

const (
	TokenParserTypeDefault TokenParserType = iota
	TokenParserTypeHarmony
)

type TokenParser struct {
	messageHandler MessageHandler
	parserEngine   ParserInternals
	toolParser     ToolParser
	lastToken      string
	tokenRepeat    int
	repeatLimit    int
}

const defaultTokenRepeatLimit = 30

type MessageHandler interface {
    AddContent(token string) (content, thinking string, toolContent string)
}

type ParserInternals interface {
    AddImplicitStartOrPrefill(prefillString string)
}

type ToolParser interface {
	Add(token string)
	Drain() (toolName *string, toolContent string)
}

// Default implementation for the TokenParser interface as a no-op passthrough
type defaultMessageHandler struct{}

func (defaultMessageHandler) AddContent(token string) (string, string, string) {
	return token, "", ""
}

type defaultEngine struct{}

func (defaultEngine) AddImplicitStartOrPrefill(prefillString string) {}

type defaultToolParser struct{}

func (defaultToolParser) Add(token string) {}

func (defaultToolParser) Drain() (*string, string) { return nil, "" }

// harmonyMessageHandlerAdapter adapts HarmonyMessageHandler to the MessageHandler interface
// by binding a per-request ToolParser instance.
type harmonyMessageHandlerAdapter struct {
    h  *harmony.HarmonyMessageHandler
    tp *harmony.HarmonyToolCallAccumulator
}

func (a harmonyMessageHandlerAdapter) AddContent(token string) (string, string, string) {
    return a.h.AddContent(token, a.tp)
}

type harmonyToolParserAdapter struct{ tp *harmony.HarmonyToolCallAccumulator }

func (a harmonyToolParserAdapter) Add(token string)                 { a.tp.Add(token) }
func (a harmonyToolParserAdapter) Drain() (*string, string)         { return a.tp.Drain() }

func NewTokenParser(parserType TokenParserType, prefillString string) TokenParser {
    switch parserType {
    case TokenParserTypeHarmony:
        harmonyMessageHandler := harmony.NewHarmonyMessageHandler()
        // Backward-compat: interpret prefillString as a literal pre-seeded tag when provided
        if strings.TrimSpace(prefillString) != "" {
            harmonyMessageHandler.HarmonyParser.AddContent(prefillString)
        } else {
            harmonyMessageHandler.HarmonyParser.AddImplicitStart()
        }
        tp := harmonyMessageHandler.CreateToolParser()
        return TokenParser{
            messageHandler: harmonyMessageHandlerAdapter{h: harmonyMessageHandler, tp: tp},
            parserEngine:   harmonyMessageHandler.HarmonyParser,
            toolParser:     harmonyToolParserAdapter{tp: tp},
            repeatLimit:    defaultTokenRepeatLimit,
        }

	default:
		return TokenParser{
			messageHandler: defaultMessageHandler{},
			parserEngine:   defaultEngine{},
			toolParser:     defaultToolParser{},
			repeatLimit:    30,
		}
	}
}

func (p *TokenParser) AddContent(token string) (string, string, error) {
	if p.repeatLimitReached(token) {
		return "", "", errors.New("token repeat limit reached")
	}
	content, thinking, toolContent := p.messageHandler.AddContent(token)
	p.toolParser.Add(toolContent)
	return content, thinking, nil
}

// repeatLimitReached updates repeat counters and returns true if the repeat limit is reached.
func (p *TokenParser) repeatLimitReached(token string) bool {
	if p == nil {
		return false
	}
	trimmed := strings.TrimSpace(token)
	if trimmed == p.lastToken {
		p.tokenRepeat++
	} else {
		p.tokenRepeat = 0
	}
	p.lastToken = trimmed

	return p.tokenRepeat >= p.repeatLimit
}

// TODO: update to work with multiple toolcalls - unmarshalling should also happen on parser level
func (p *TokenParser) Drain() []api.ToolCall {
	toolName, toolContent := p.toolParser.Drain()
	if toolName != nil {
		*toolName = strings.TrimPrefix(*toolName, "functions.")
		var args api.ToolCallFunctionArguments
		if err := json.Unmarshal([]byte(toolContent), &args); err != nil {
			return nil
		}
		return []api.ToolCall{
			{
				Function: api.ToolCallFunction{
					Name:      *toolName,
					Arguments: args,
				},
			},
		}
	}
	return nil
}
