package compat

import "strings"

type EndpointKind string

const (
	EndpointUnknown         EndpointKind = "unknown"
	EndpointResponses       EndpointKind = "responses"
	EndpointChatCompletions EndpointKind = "chat_completions"
	EndpointCompletions     EndpointKind = "completions"
	EndpointMessages        EndpointKind = "messages"
	EndpointEmbeddings      EndpointKind = "embeddings"
	EndpointImages          EndpointKind = "images"
	EndpointAudioSpeech     EndpointKind = "audio_speech"
	EndpointAudioVoice      EndpointKind = "audio_voice"
	EndpointAudioTranscribe EndpointKind = "audio_transcribe"
	EndpointRerank          EndpointKind = "rerank"
	EndpointInfill          EndpointKind = "infill"
	EndpointCompletion      EndpointKind = "completion"
)

func Route(path string) EndpointKind {
	switch {
	case strings.HasPrefix(path, "/v1/responses"):
		return EndpointResponses
	case strings.HasPrefix(path, "/v1/chat/completions"):
		return EndpointChatCompletions
	case strings.HasPrefix(path, "/v1/completions"):
		return EndpointCompletions
	case strings.HasPrefix(path, "/v1/messages"):
		return EndpointMessages
	case strings.HasPrefix(path, "/v1/embeddings"):
		return EndpointEmbeddings
	case strings.HasPrefix(path, "/v1/images/"):
		return EndpointImages
	case strings.HasPrefix(path, "/v1/audio/speech"):
		return EndpointAudioSpeech
	case strings.HasPrefix(path, "/v1/audio/voices"):
		return EndpointAudioVoice
	case strings.HasPrefix(path, "/v1/audio/transcriptions"):
		return EndpointAudioTranscribe
	case strings.HasPrefix(path, "/v1/rerank"), strings.HasPrefix(path, "/v1/reranking"), strings.HasPrefix(path, "/rerank"), strings.HasPrefix(path, "/reranking"):
		return EndpointRerank
	case strings.HasPrefix(path, "/infill"):
		return EndpointInfill
	case strings.HasPrefix(path, "/completion"):
		return EndpointCompletion
	default:
		return EndpointUnknown
	}
}

func IsInferencePath(path string) bool {
	return Route(path) != EndpointUnknown
}

func IsJSONBodyEndpoint(kind EndpointKind) bool {
	switch kind {
	case EndpointResponses,
		EndpointChatCompletions,
		EndpointCompletions,
		EndpointMessages,
		EndpointEmbeddings,
		EndpointImages,
		EndpointAudioSpeech,
		EndpointRerank,
		EndpointInfill,
		EndpointCompletion:
		return true
	default:
		return false
	}
}
